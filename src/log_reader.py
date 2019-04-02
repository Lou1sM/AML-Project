import matplotlib.pyplot as plt
import sys

def get_train_val_scores(filename):
    lines = [line for line in open(filename).readlines() if "Epoch loss" in line]
    train_losses = [float(line[17:25]) for line in lines]
    val_losses = [float(line[-9:-1]) for line in lines]
    return train_losses, val_losses

def plot_losses(train_losses, val_losses=None):
    #plt.axis([None, None, 0, 9])
    plt.ylim(0,6)
    print(train_losses)
    print(val_losses)
    #train_losses = list(range(13))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')

    plt.legend()
    plt.title('Batch size: 64    Pool size: 4')
    plt.savefig('perf_graph.png')
    plt.show()
 
if __name__ == "__main__":
    log_file_name = sys.argv[1]
    train_loss_scores, val_loss_scores = get_train_val_scores(log_file_name)
    plot_losses(train_losses=train_loss_scores, val_losses=val_loss_scores)
