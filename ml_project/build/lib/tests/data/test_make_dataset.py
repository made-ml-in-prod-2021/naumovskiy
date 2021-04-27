from ml_example.data.make_dataset import read_data, split_train_val_data, generate_fake_dataset
from ml_example.enities import SplittingParams


def test_read_data(dataset_path: str, target_col: str):
    data = read_data(dataset_path)
    assert len(data) > 10
    assert target_col in data.keys()


def test_split_train_val_data(dataset_path: str):
    val_size = 0.2
    splitting_params = SplittingParams(random_state=42, val_size=val_size,)
    data = read_data(dataset_path)
    train, val = split_train_val_data(data, splitting_params)
    assert train.shape[0] > 10
    assert val.shape[0] > 10


def test_generate_fake_dataset(size_dataset: int):
    data = generate_fake_dataset(size_dataset)
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
               'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    assert size_dataset == data.shape[0]
    assert len(set(columns) - set(data.columns)) == 0
