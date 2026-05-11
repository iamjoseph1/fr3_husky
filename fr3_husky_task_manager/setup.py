from setuptools import find_packages, setup

package_name = 'fr3_husky_task_manager'

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
    maintainer='yuminlim',
    maintainer_email='ckrgksakdmac@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'apple_vision_pro = fr3_husky_task_manager.apple_vision_pro:main',
            'sa_apple_vision_pro = fr3_husky_task_manager.sa_apple_vision_pro:main',
            'husky_pedal = fr3_husky_task_manager.husky_pedal:main',
            'move_to_joint = fr3_husky_task_manager.move_to_joint:main',
            'task_move = fr3_husky_task_manager.task_move:main',
            'gripper_move = fr3_husky_task_manager.gripper_move:main',
        ],
    },
)
