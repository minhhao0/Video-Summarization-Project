from bertopic import BERTopic
from underthesea import sent_tokenize
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
class TopicModel:
    def __init__(self,model_path):
        self.represent_model = SentenceTransformer(model_path)
        print('Load topic model successfully')
        self.topic_model = BERTopic(
            embedding_model=self.represent_model,
            verbose=True,
            language='multilingual'
        )
    def get_topic(self,paragraph):
        sentences = sent_tokenize(paragraph)
        segmented_sentences = [ViTokenizer.tokenize(sentence) for sentence in sentences]
        topics, prob = self.topic_model.fit_transform(segmented_sentences)
        freq = self.topic_model.get_topic_info()
        if freq.shape[0] != 1:
            topic = freq.iloc[1]
        else:
            topic = freq.iloc[0]
        final_topic = topic['Representative_Docs'][0]
        return final_topic