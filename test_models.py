from models.simple_model import test_simple_model
from models.unet_model import test_densedepth


if __name__ == '__main__':
    print("Testing...")
    test_simple_model()
    test_densedepth()
    print("All test passed")
