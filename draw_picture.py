import os
import matplotlib.pyplot as plt

# sgd adam sam_sgd sam_adam esam_sgd esam_adam looksam_sgd looksam_adam dynamicsam_sgd dynamicsam_adam
METHODS_SGD = ['sgd', 'sam_sgd', 'esam_sgd', 'looksam_sgd', 'dynamicsam_sgd']
METHODS_ADAM = ['adam', 'sam_adam', 'esam_adam', 'looksam_adam', 'dynamicsam_adam']

MODEL = 'resnet18' # resnet18 resnet34
BATCH_SIZE = 256 # 64 128 256
EPOCHS = 100
DRAW_STEP = 10

RESULT_DIR = f'result/{MODEL}/{BATCH_SIZE}'
PICTURE_DIR = f'picture_comparison/{MODEL}/{BATCH_SIZE}'
SGD_PICTURE_PATH = f'{PICTURE_DIR}/SGD'
ADAM_PICTURE_PATH = f'{PICTURE_DIR}/ADAM'

if not os.path.exists(SGD_PICTURE_PATH):
    os.makedirs(SGD_PICTURE_PATH)
if not os.path.exists(ADAM_PICTURE_PATH):
    os.makedirs(ADAM_PICTURE_PATH)

def draw(methods, picture_path):
    colors = ['blue', 'green', 'red', 'yellow', 'orange', 'purple', 'brown', 'pink']

    results = {
        'train_loss': {},
        'test_loss': {},
        'train_accuracy': {},
        'test_accuracy': {}
    }

    # 文件组织结构
    """
    train_losses: [4.285000466570562, 3.745828174814886, ...]
    test_losses: [3.926690699005127, 3.5972017642974854, ...]
    train_accuracies: [12.132, 17.154, ...]
    test_accuracies: [11.25, 16.71, ...]
    """

    for method in methods:
        result_path = f'{RESULT_DIR}/training_results_{method}.txt'
        if not os.path.exists(result_path):
            raise FileNotFoundError(f'{result_path} not found')
        with open(result_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('train_losses'):
                    results['train_loss'][method] = eval(line.split(':')[1].strip())
                elif line.startswith('test_losses'):
                    results['test_loss'][method] = eval(line.split(':')[1].strip())
                elif line.startswith('train_accuracies'):
                    results['train_accuracy'][method] = eval(line.split(':')[1].strip())
                elif line.startswith('test_accuracies'):
                    results['test_accuracy'][method] = eval(line.split(':')[1].strip())

    # 分别绘制不同优化器的 loss 和 accuracy 的对比图
    figsize = (12, 6)

    # Train Losses
    plt.figure(figsize=figsize)
    # plt.subplot(2, 2, 1)
    for method in methods:
        plt.plot(results['train_loss'][method], label=method, color=colors[methods.index(method)], linestyle='--' if methods.index(method) == 0 else '-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.xticks(range(0, EPOCHS, DRAW_STEP))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{picture_path}/train_loss.png")

    # Test Losses
    plt.figure(figsize=figsize)
    # plt.subplot(2, 2, 2)
    for method in methods:
        plt.plot(results['test_loss'][method], label=method, color=colors[methods.index(method)], linestyle='--' if methods.index(method) == 0 else '-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    plt.xticks(range(0, EPOCHS, DRAW_STEP))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{picture_path}/test_loss.png")

    # Train Accuracies
    plt.figure(figsize=figsize)
    # plt.subplot(2, 2, 3)
    for method in methods:
        plt.plot(results['train_accuracy'][method], label=method, color=colors[methods.index(method)], linestyle='--' if methods.index(method) == 0 else '-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train Accuracy')
    plt.xticks(range(0, EPOCHS, DRAW_STEP))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{picture_path}/train_accuracy.png")

    # Test Accuracies
    plt.figure(figsize=figsize)
    # plt.subplot(2, 2, 4)
    for method in methods:
        plt.plot(results['test_accuracy'][method], label=method, color=colors[methods.index(method)], linestyle='--' if methods.index(method) == 0 else '-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.xticks(range(0, EPOCHS, DRAW_STEP))
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{picture_path}/test_accuracy.png")


def main():
    draw(METHODS_SGD, SGD_PICTURE_PATH)
    draw(METHODS_ADAM, ADAM_PICTURE_PATH)

if __name__ == '__main__':
    main()
