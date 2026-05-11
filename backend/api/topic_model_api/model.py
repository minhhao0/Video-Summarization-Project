import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from bertopic import BERTopic
from underthesea import sent_tokenize
from pyvi import ViTokenizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

class TopicModel:
    def __init__(self,model_path):
        self.represent_model = SentenceTransformer(model_path)
        print('Load topic model successfully')

    def get_topic(self,paragraph):
      try:
        sentences = sent_tokenize(paragraph)
        segmented_sentences = [ViTokenizer.tokenize(sentence) for sentence in sentences]
        n = len(sentences)
        n_neighbors = max(2, min(15, n // 2))
        min_cluster_size = max(2, n // 5)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=min(2, n - 1),
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            prediction_data=True
        )
        topic_model = BERTopic(
            embedding_model=self.represent_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            verbose=True,
            language='multilingual'
        )
        topics, prob = topic_model.fit_transform(segmented_sentences)
        freq = topic_model.get_topic_info()
        if freq.shape[0] != 1:
            topic = freq.iloc[1]
        else:
            topic = freq.iloc[0]
        final_topic = ''.join(topic['Representative_Docs'])
        return final_topic.replace('_',' ')
      except:
          return "Không xác định được chủ để cho văn bản trên"
# if __name__=='__main__':
#     topic_model=TopicModel('D:\\Video-Summarization-Project\\model\\vietnamese-embedding')
#     p='''
#     tại nhật bản thì tài sản do người cao tuổi bị mất trí nhớ nắm giữ có thể vượt quá năm trăm nghìn tỷ yên tương đương với ba nghìn ba trăm tỷ đô la mỹ và năm hai nghìn không trăm ba mươi tài sản đó thì có thể là những bất động sản cổ phiếu trái phiếu tiền trong tài khoản ngân hàng có nguy cơ bị đóng băng tác động nhiều chiều đến thị trường phản ánh của phóng viên quang hưng thường trú đài truyền hình việt nam tại nhật bản. năm hai nghìn không trăm hai mươi tổng tài sản do người mắc chứng mất trí nhớ nắm giữ đã là một trăm tám mươi bảy nghìn tỷ yên đến năm hai nghìn không trăm ba mươi con số này dự kiến là hai trăm bốn mươi hai nghìn tỷ yên và nếu tính cả người cao tuổi bị suy giảm trí nhớ ở giai đoạn đầu thì tổng tài sản có thể lên tới năm trăm ba mươi ba nghìn tỷ yên tương đương với ba nghìn năm trăm năm mươi tỷ đô la mỹ nếu ngân hàng hoặc tổ chức tài chính xác định một người thiếu năng lực đưa ra quyết định tài khoản và giao dịch của họ có thể sẽ bị đọc băng. xử lý tài sản liên quan đến bất động sản cổ phiếu trái phiếu sẽ rất phức tạp và cần phải chỉ định người giám hộ hợp pháp nhiều trường hợp khó khăn trong việc chỉ định người giám hộ việc đóng băng này cũng ảnh hưởng đến thị trường tiêu dùng các khoản chi trả y tế viện nghiên cứu nhật bản ước tính mức tiêu dùng của người cao tuổi mắc chứng mất trí nhớ năm hai nghìn không trăm hai lăm đạt mười bốn phẩy bảy nghìn tỷ yên tức là khoảng chín mươi tám tỷ đô la mỹ theo sách chẳng của chính phủ nhật bản năm hai nghìn không trăm hai lăm số người suy giảm trí nhớ đạt năm phẩy hai triệu người. và sẽ tăng mười một phần trăm vào năm hai nghìn không trăm ba mươi nhật bản đang sửa đổi luật dân sự để đơn giản hoá hoạt động giám hộ tự nguyện và lập các quỹ tín thác để quản lý số tài sản khổng lồ này tài sản của người mất trí nhớ hoặc suy giảm nhận thức nghiêm trọng thì việc xử lý sẽ rất mất thời gian nhất là khi họ muốn chuyên đổi thành tiền để chi trả cho các dịch vụ y tế chữa bệnh tình trạng này có tác động trên một quy mô rộng hơn đó là nền kinh tế của người cao tuổi hoàng hưng phóng viên thường trú đài trình việt nam. đưa tin từ tokyo nhật bản.
#     '''
#     final_topic=topic_model.get_topic(p)
#     summary_model=Summarizer('D:\\Video-Summarization-Project\\model\\mt5')
#     print(summary_model.run(final_topic))