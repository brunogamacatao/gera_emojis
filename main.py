import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

CANAIS  = 1
LARGURA = 16
ALTURA  = 16

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

class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.entrada = nn.Linear(CANAIS * LARGURA * ALTURA, 4 * CANAIS * LARGURA * ALTURA) 
    self.oculta1 = nn.Linear(4 * CANAIS * LARGURA * ALTURA, 8 * CANAIS * LARGURA * ALTURA)
    self.oculta2 = nn.Linear(8 * CANAIS * LARGURA * ALTURA, 4 * CANAIS * LARGURA * ALTURA)
    self.saida   = nn.Linear(4 * CANAIS * LARGURA * ALTURA, CANAIS * LARGURA * ALTURA) 
    self.activation = nn.Sigmoid()

  def forward(self, x):
    x = x.view(CANAIS * LARGURA * ALTURA)
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
    self.linear1 = nn.Linear(CANAIS * LARGURA * ALTURA, 16 * CANAIS * LARGURA * ALTURA)
    self.linear2 = nn.Linear(16 * CANAIS * LARGURA * ALTURA, 2 * LARGURA * ALTURA)
    self.linear3 = nn.Linear(2 * LARGURA * ALTURA, 1)
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

def treina(epochs = 100):
  print('Vou começar o treinamento ...')
  for epoch in range(epochs):
    print(f'Treinando epoch {epoch + 1} ...')
    for batch, (true_data, _) in enumerate(train_loader): # ingora o label do loader
      true_labels  = torch.ones(1, 1) # as imagens são consideradas entradas reais
      false_labels = torch.zeros(1, 1) # label para as imagens geradas
      
      # Treinamento do generator
      generator_optimizer.zero_grad() # Zera os gradientes do generator

      # Criamos um ruído aleatório para o gerador
      noise = torch.randn(1, CANAIS, LARGURA, ALTURA)
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
    noise = torch.randn(1, CANAIS, LARGURA, ALTURA)
    generated = generator(noise)
    save_image(generated, f'./samples/sample_{epoch}.png')

if __name__ == '__main__':
  treina(500)