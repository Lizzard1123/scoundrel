from setuptools import setup, find_packages

setup(
    name="scoundrel",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich>=13.0.0",
    ],
    entry_points={
        'console_scripts': [
            'play=main:main',
            'mcts=scoundrel.rl.mcts.viewer:main',
            'mcts-plot=scoundrel.rl.mcts.plot_episode:main',
            'mcts-eval=scoundrel.rl.mcts.eval:main',
            'mcts-collect=scoundrel.rl.mcts.collect:main',
            'mcts-evaluate-collection=scoundrel.rl.mcts.evaluate_collection:main',
        ],
    },
)