# ERLDK
multimodal emotion recognition model for conversational videos based on reinforcement learning and domain knowledge

This is the open access code for the paper: Real-Time Video Emotion Recognition based on Reinforcement Learning and Domain Knowledge, which has been published online on April 12, 2021. This paper can be download from https://ieeexplore.ieee.org/document/9400391.

Here's the abstract.

Multimodal emotion recognition in conversational videos (ERC) develops rapidly in recent years. To fully extract the relative context from video clips, most studies build their models on the entire dialogues which make them lack of real-time ERC ability. Different from related researches, a novel multimodal emotion recognition model for conversational videos based on reinforcement learning and domain knowledge (ERLDK) is proposed in this paper. In ERLDK, the reinforcement learning algorithm is introduced to conduct real-time ERC with the occurrence of conversations. The collection of history utterances is composed as an emotion-pair which represents the multimodal context of the following utterance to be recognized. Dueling deep-Q-network (DDQN) based on gated recurrent unit (GRU) layers is designed to learn the correct action from the alternative emotion categories. Domain knowledge is extracted from public dataset based on the former information of emotion-pairs. The extracted domain knowledge is used to revise the results from the RL module and is transformed into other dataset to examine the rationality. The experimental results on datasets show that ERLDK achieves the state-of-the-art results on weighted average and most of the specific emotion categories.

How to use this repository:

read.py:
dataloader.py:
knowledge.py:
dqn_env_iemocap.py:
dqn_env_meld.py:
dialogue_level_test.py:
