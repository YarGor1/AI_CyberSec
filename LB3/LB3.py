import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

def show_img(x_0, y_0):
    plt.imshow(x_0, cmap='gray') 
    plt.title(f"Label: {y_0}")
    plt.show()

# Створення наборів для тренування
train_set = torchvision.datasets.MNIST("./mnist/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./mnist/", train=False, download=True)
#print(train_set)
#print(valid_set)

x_0, y_0 = train_set[0]
#show_img(x_0, y_0)
#print(type(x_0))

# Перетворення на тензори
trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)
#print(x_0_tensor.dtype)
#print(x_0_tensor.min())
#print(x_0_tensor.max())
#print(x_0_tensor.size())
#print(x_0_tensor)
#print(x_0_tensor.device)

image = F.to_pil_image(x_0_tensor)
#show_img(image, y_0)


trans = transforms.Compose([transforms.ToTensor()])
train_set.transform = trans
valid_set.transform = trans

batch_size = 32

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)


test_matrix = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
#print(test_matrix)
#print(nn.Flatten()(test_matrix))
batch_test_matrix = test_matrix[None, :]
#print(batch_test_matrix)
#print(nn.Flatten()(batch_test_matrix))


input_size = 1 * 28 * 28
n_classes = 10

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
]
print(layers)

model = nn.Sequential(*layers)
print(model)

loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train(a, b):
    loss = 0
    accuracy = 0

    model.train()
    for x, y in train_loader:
        #x, y = x.to(device), y.to(device)
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        accuracy += get_batch_accuracy(output, y, train_N)
    a.append(loss)
    b.append(accuracy)
    print('Train - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

    
def validate(a, b):
    loss = 0
    accuracy = 0

    model.eval()
    with torch.no_grad():
        for x, y in valid_loader:
            #x, y = x.to(device), y.to(device)
            output = model(x)

            loss += loss_function(output, y).item()
            accuracy += get_batch_accuracy(output, y, valid_N)
    a.append(loss)
    b.append(accuracy)
    print('Valid - Loss: {:.4f} Accuracy: {:.4f}'.format(loss, accuracy))

trls, trac, vals, vaac = [], [], [], []
epochs = 5

for epoch in range(epochs):
    print('Epoch: {}'.format(epoch))
    train(trls, trac)
    validate(vals, vaac)

#print(trls, trac, vals, vaac)
prediction = model(x_0_tensor)
print(prediction)
print(prediction.argmax(dim=1))

#show_img(x_0, y_0)


plt.subplot(1, 2, 1)
plt.plot(trls, label='Train Loss')
plt.plot(vals, label='Valid Loss')
plt.legend()
plt.title("Loss Progression")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(trac, label='Train Accuracy')
plt.plot(vaac, label='Valid Accuracy')
plt.legend()
plt.title("Accuracy Progression")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.show()


