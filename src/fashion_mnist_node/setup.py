from setuptools import find_packages, setup

package_name = 'fashion_mnist_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rex',
    maintainer_email='rex@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'fashion_detection_publisher = fashion_mnist_node.fashion_detection_publisher:main',
            'fashion_detection_subscriber = fashion_mnist_node.fashion_detection_subscriber:main',
        ],
    },
)
