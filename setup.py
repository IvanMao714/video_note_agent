from setuptools import setup, find_packages

setup(
    name="opencourse-agent",
    version="0.1.0",
    description="OpenCourse Agent for note generation from slides and videos",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # 依赖项会在运行时从 requirements.txt 或其他地方读取
        # 这里只定义包结构
    ],
)






