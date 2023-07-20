import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Sample code for training a simple classifier on the iris dataset
# Loading the data, scaling, and splitting between train/test
data = load_iris()
X = data['data']
y = data['target']
X_scaled = StandardScaler().fit_transform(X)

# Using sklearn's train_test_split instead of torch's random_split for convenience
# since labels are already declared
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=46)

# Converting to tensors for torch
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

# Defining the model architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x

# Specifying the model, loss function, and optimizer
model = NeuralNetwork(input_dim=X_train.shape[1], num_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training the model, storing loss and accuracy for each epoch
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
num_epochs = 100
for epoch in range(num_epochs):
    # Specifying that the model is in training mode
    model.train()

    # Clearing out the gradient from the last step of loss.backward
    optimizer.zero_grad()

    # Forward feeding the model to generate predictions
    y_pred = model.forward(X_train)

    # Calculating the training
    loss = criterion(y_pred, y_train)
    train_losses.append(loss.item())

    # Calculating the gradients w/ backprop and updating the weights
    loss.backward()
    optimizer.step()

    # Specifying that the model is in evaluation mode to gather val loss and train/test accuracy
    model.eval()
    with torch.no_grad():
        # Generating predictions on the train/test sets
        y_train_pred = model.forward(X_train)
        y_test_pred = model.forward(X_test)

        # Collecting the test loss
        test_loss = criterion(y_test_pred, y_test)
        test_losses.append(test_loss.item())

        # Collecting the train accuracy
        y_train_pred = torch.argmax(y_train_pred, dim=1)
        accuracy = torch.sum(y_train_pred == y_train).item() / len(y_train_pred)
        train_accuracies.append(accuracy)

        # Collecting the test accuracy
        y_test_pred = torch.argmax(y_test_pred, dim=1)
        test_accuracy = torch.sum(y_test_pred == y_test).item() / len(y_test_pred)
        test_accuracies.append(test_accuracy)

    # Reporting reuslts every 5 epochs
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {loss.item():.4} | Train Accuracy: {accuracy:.4} | Test Loss: {test_loss.item():.4} | Test Accuracy: {test_accuracy:.4}")


### Early stopping
# Taken from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch/71999355#71999355
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Implementing the above early stopping class when training
early_stopper = EarlyStopper(patience=3, min_delta=10)
for epoch in np.arange(n_epochs):
    train_loss = train_one_epoch(model, train_loader)
    validation_loss = validate_one_epoch(model, validation_loader)
    if early_stopper.early_stop(validation_loss):
        print(f'Early stopping at epoch {epoch+1} with val loss of {validation_loss}')
        break


### Using the data loader for mini batches while training
# Assuming the data is in a numpy array for this snippet
from torch.utils.data import TensorDataset, DataLoader
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

n_epochs = 50
for epoch in range(n_epochs):
    # Calculate the loss at the beginning of the epoch
    optimizer.zero_grad()
    y_pred = model.forward(X)
    loss = criterion(y_pred, y)
    losses.append(loss)
    # Update the model params within the mini batches
    for X_batch, y_batch in dataloader:
        y_pred = model.forward(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    

### Dataset for NumPy arrays
from torch.utils.data import DataLoader, Dataset

class NumpyArrayDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        file_path = os.path.join(self.root_dir, file_name)
        numpy_array = np.load(file_path)
        tensor_array = torch.from_numpy(numpy_array)
        return tensor_array

dataset = NumpyArrayDataset(path_to_numpy_arrays)
