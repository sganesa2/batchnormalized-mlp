from data.dataset import NgramDataset
from model.train import BatchNormalizedMLP
from model.inference import run_batchnormlized_mlp

def main()->list[str]:
    #Dataset creation
    dataset = NgramDataset(3, "dataset.txt", 25626,3204,3203)
    x,y = dataset.get_complete_dataset()
    x_train,y_train = dataset.trainset

    #Model training
    model = BatchNormalizedMLP(200, 3, feature_dims=10)
    model.minibatch_gradient_descent(32, x_train, y_train, 1000, 0.1,0)

    #Inference
    words = run_batchnormlized_mlp(10, model, 'emm')
    return words

if __name__=="__main__":
    words = main()
    print(words)