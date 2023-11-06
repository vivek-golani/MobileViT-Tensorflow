from data.BaseTempDataset import create_temp_dataset

def create_dataset(dataset_kwargs):
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs['dataset']

    if dataset_name == 'temp':
        dataset = create_temp_dataset(**dataset_kwargs)
    else:
        raise ValueError(f'Dataset {dataset_name} not known')
    
    return dataset
    