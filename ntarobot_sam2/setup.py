from setuptools import find_packages, setup
import os
import glob

package_name = 'ntarobot_sam2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join("share", package_name, "assets"),
            glob.glob("assets/*.png") + glob.glob("assets/*.jpg"),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='eliodavid',
    maintainer_email='eliodavid@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_sam_server = scripts.sam2_server_node:main',
            'node_sam_client = scripts.sam2_client_node:main',
        ],
    },
)
