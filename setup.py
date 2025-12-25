from setuptools import setup, find_packages

setup(
    name="scoundrel",
    version="0.1.0",
    packages=find_packages(),
    py_modules=['main'],
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
            'policy-large-train=scoundrel.rl.alpha_scoundrel.policy.policy_large.train:main',
            'policy-large-viewer=scoundrel.rl.alpha_scoundrel.policy.policy_large.viewer:main',
            'policy-small-train=scoundrel.rl.alpha_scoundrel.policy.policy_small.train:main',
            'policy-small-viewer=scoundrel.rl.alpha_scoundrel.policy.policy_small.viewer:main',
            'value-large-train=scoundrel.rl.alpha_scoundrel.value.value_large.train:main',
            'value-large-viewer=scoundrel.rl.alpha_scoundrel.value.value_large.viewer:main',
        ],
    },
)