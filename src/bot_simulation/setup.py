import os
from setuptools import find_packages, setup
from glob import glob

package_name = 'bot_simulation'

# Include all files in model subdirectories
model_files = [(os.path.join('share', package_name, root), 
                [os.path.join(root, f) for f in files])
               for root, _, files in os.walk('model') if files]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ] + model_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='himanshu',
    maintainer_email='himanshubaldeva@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
