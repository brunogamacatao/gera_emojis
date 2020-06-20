import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

CANAIS  = 1
LARGURA = 32
ALTURA  = 32

ARQUIVO_REDE = 'emojis.pth'
DATA_PATH    = 'data'
BATCH_SIZE   = 1 # vamos trabalhar com 1 imagem por vez

transformacoes = transforms.Compose([
  transforms.Grayscale(num_output_channels = 1),
  transforms.Resize([LARGURA, ALTURA]),
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5))
])

train_dataset = torchvision.datasets.ImageFolder(
  root = DATA_PATH,
  transform = transformacoes
)

train_loader = torch.utils.data.DataLoader(
  train_dataset,
  batch_size=BATCH_SIZE,
  num_workers=2,
  shuffle=True
)

QTD_IMAGENS_REAIS = 7

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.entrada = nn.Linear(QTD_IMAGENS_REAIS, 120) 
    self.oculta1 = nn.Linear(120, 256)
    self.oculta2 = nn.Linear(256, CANAIS * LARGURA * ALTURA)
    self.saida   = nn.Linear(CANAIS * LARGURA * ALTURA, CANAIS * LARGURA * ALTURA) 
    self.activation = nn.Sigmoid()

  def forward(self, x):
    #x = x.view(CANAIS * LARGURA * ALTURA)
    x = self.activation(self.entrada(x))
    x = self.activation(self.oculta1(x))
    x = self.activation(self.oculta2(x))
    x = self.activation(self.saida(x))
    return x.view(1, CANAIS, LARGURA, ALTURA)

class DiscriminatorConv(nn.Module):
  def __init__(self):
    super(DiscriminatorConv, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5) # canais, qtd filtros, kernel
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.linear1 = nn.Linear(1024, 512)
    self.linear2 = nn.Linear(512, 128)
    self.linear3 = nn.Linear(128, 1)
    #self.pool = nn.MaxPool2d(2, 2)
    self.activation = nn.Sigmoid() # Passa a saída por uma sigmoide (0, 1)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(-1, 1024)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.activation(self.linear3(x))
    return x

class DiscriminatorLinear(nn.Module):
  def __init__(self):
    super(DiscriminatorLinear, self).__init__()
    self.linear1 = nn.Linear(CANAIS * LARGURA * ALTURA, CANAIS * LARGURA * ALTURA)
    self.linear2 = nn.Linear(CANAIS * LARGURA * ALTURA, LARGURA * ALTURA // 2)
    self.linear3 = nn.Linear(LARGURA * ALTURA // 2, 1)
    self.activation = nn.Sigmoid()

  def forward(self, x):
    x = x.view(CANAIS * LARGURA * ALTURA)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.activation(self.linear3(x))
    return x


# Vamos começar o treinamento !
print('Instanciando os modelos ...')
generator = Generator()
discriminator = DiscriminatorLinear()

print('Instanciando os otimizadores ...')
# Configuração dos otimizadores
loss = nn.BCELoss() # Binary Cross Entropy Loss
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001) # otimizador do generator
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001) # otimizador do discriminator

def cria_tensor_de_entrada(n, total = QTD_IMAGENS_REAIS):
  # se n for 0 e total 4, saída: 1000
  # se n for 2 e total 4, saída: 0010
  # se n for 3 e total 4, saída: 0001
  zeros_antes  = n
  zeros_depois = total - n - 1
  n_str = '0' * zeros_antes + '1' + '0' * zeros_depois
  n_list = [int(x) for x in list(n_str)]
  return torch.Tensor(n_list).float()

def treina(epochs = 100):
  print('Vou começar o treinamento ...')
  for epoch in range(epochs):
    print(f'Treinando epoch {epoch + 1} ...')
    for batch, (true_data, label) in enumerate(train_loader): # ingora o label do loader
      true_labels  = torch.ones(1, 1) # as imagens são consideradas entradas reais
      false_labels = torch.zeros(1, 1) # label para as imagens geradas
      
      # Treinamento do generator
      generator_optimizer.zero_grad() # Zera os gradientes do generator

      # Criamos um ruído aleatório para o gerador
      noise = cria_tensor_de_entrada(int(label.item()))  #torch.randn(1, CANAIS, LARGURA, ALTURA)
      generated_data = generator(noise) # Obtemos a saída do gerador

      # Otimização do generator
      generator_discriminator_out = discriminator(generated_data)     # passa o valor gerado no generator para o discriminator
      generator_loss = loss(generator_discriminator_out, true_labels) # calcula o erro do discriminador
      generator_loss.backward()  # calcula os gradientes do generator
      generator_optimizer.step() # atualiza os pesos do generator

      # Treinamos o discriminator
      # Passo 1: Treinamos a capacidade do discriminator de identificar dados reais
      discriminator_optimizer.zero_grad() # zeramos o gradiente do discriminator
      true_discriminator_out = discriminator(true_data) # passamos dados reais
      true_discriminator_loss = loss(true_discriminator_out, true_labels) # erro para identificar dados reais

      # Passo 2: Treinamos a capacidade do discriminator de identificar dados falsos (gerados)
      generator_discriminator_out = discriminator(generated_data.detach()) # o detach é para evitar atualizar o gradiente do generator
      generator_discriminator_loss = loss(generator_discriminator_out, false_labels) # erro para identificar dados falsos

      # O erro final é a média entre a capacidade de identificar dados reais e a capacidade de identificar dados falsos
      discriminator_loss = (true_discriminator_loss + generator_discriminator_loss) / 2
      discriminator_loss.backward() # calcula os gradientes do discriminator
      discriminator_optimizer.step() # atualiza os pesos do discriminator
    print('\tGerando 1 imagem...')
    generate_image(epoch)

def generate_image(epoch):
  with torch.no_grad():
    for i in range(QTD_IMAGENS_REAIS):
      noise = cria_tensor_de_entrada(i) #torch.randn(1, CANAIS, LARGURA, ALTURA)
      generated = generator(noise)
      save_image(generated, f'./samples/emoji_e{epoch}_t{i}.png')

if __name__ == '__main__':
  treina(500)