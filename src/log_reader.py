import matplotlib.pyplot as plt
import sys
import os

def get_train_val_scores(filename):
    filename_lines = open(filename).readlines()
    train_losses = [float(line[17:25]) for line in filename_lines if "Epoch loss" in line]
    val_losses = [float(line[23:30]) for line in filename_lines if "Epoch validation loss" in line]
    return train_losses, val_losses

def plot_losses(train_losses, val_losses=None, title='', filepath=None): 
    #plt.axis([None, None, 0, 9])
    plt.ylim(0,6)
    #print(train_losses)
    #print(val_losses)
    #print(filepath)
    #train_losses = list(range(13))
    plt.plot(train_losses, label='Train loss')
    plt.plot(val_losses, label='Validation loss')

    plt.legend()
    plt.title(title)
    if filepath == None:
        filepath = 'train_val_loss.png'
    plt.savefig(filepath)
    plt.show()
 
if __name__ == "__main__":
    log_file_path = sys.argv[1]
    try:
        title = sys.argv[2]
    except IndexError:
        title = ''
    print(log_file_path)
    out_dir = os.path.dirname(log_file_path)
    train_loss_scores, val_loss_scores = get_train_val_scores(log_file_path)
    plot_losses(train_losses=train_loss_scores, val_losses=val_loss_scores, title=title)
