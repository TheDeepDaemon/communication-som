import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from population_grid import PopulationGrid
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np



def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:

            optimizer.zero_grad()

            loss = model.step(torch.flatten(images, start_dim=1), criterion)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main():

    ROWS = 5
    COLS = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    population_grid = PopulationGrid(
        rows=ROWS, cols=COLS, concept_size=28*28, hidden_size=64, language_size=32).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(population_grid.parameters(), lr=0.001)

    train(population_grid, train_loader, criterion, optimizer, epochs=5)

    for images, _ in test_loader:
        test_image = torch.flatten(images, start_dim=1)
        break

    output_grid = torch.zeros((population_grid.rows, population_grid.cols, 32))
    for i in range(ROWS):
        for j in range(COLS):
            person = population_grid.population[i][j]
            person.set_concept(test_image)
            output_grid[i, j] = person.encode_concept_to_language()

    # then do PCA on the language data
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(output_grid.detach().numpy().reshape((-1, 32)))
    transformed = transformed.reshape((ROWS, COLS, 3))
    transformed -= np.min(transformed)
    transformed /= np.max(transformed)

    # Display the matrix as an image
    plt.imshow(transformed)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
