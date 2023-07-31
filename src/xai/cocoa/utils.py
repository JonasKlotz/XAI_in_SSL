import torch

from PIL import Image

def get_black_baseline(
    explicand: torch.Tensor
) -> torch.Tensor:
    """
    Get black baseline values for a batch of explicand inputs.

    Args:
    ----
        explicand: A batch of explicands with shape `(batch_size, *)`, where `*`
            indicates the model input size for one sample.
        dataset_name: Name of the dataset where the explicands come from. This is to
            determine the channel means and standard deviations for normalization.
        normalize: Whether to normalize the black baseline values.

    Returns
    -------
        Black baseline values with shape `(batch_size, *)`, on the same device as
        `explicand`.
    """
    black_baseline = torch.zeros_like(explicand)
    return black_baseline

def create_foil_set(amount: int):
    # Generate random noise tensor
    noise = torch.randn(350, 128, 128, 3)

    # Save random noise images
    for i in range(noise.shape[0]):
        # Convert the tensor to a PIL image
        image = Image.fromarray((noise[i] * 255).byte().numpy(), mode="RGB")
        
        # Save the image
        image.save(f"./dataset/noise/image_{i}.png")

def create_corpus_set(data_path, destiny_path, num_samples=350):
    random_numbers = [random.sample(range(0,len(os.listdir(data_path)),2),350)][0]
    random_masks = [_+1 for _ in random_numbers]

    selected_elements = [os.listdir(data_path)[index] for index in random_numbers if 0 <= index < len(os.listdir(data_path))]
    selected_masks = [os.listdir(data_path)[index] for index in random_masks if 0 <= index < len(os.listdir(data_path))]

    for idx, file_name in enumerate(selected_elements):
            source_path = os.path.join(data_path, file_name)
            destination_path = os.path.join(destiny_path, file_name)
            shutil.move(source_path, destination_path)
            print("moved: " + source_path + " to: " + str(destination_path))


    for idx, file_name in enumerate(selected_masks):
            source_path = os.path.join(data_path, file_name)
            os.remove(source_path)
            print("removed: " + source_path)
