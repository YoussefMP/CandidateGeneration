# from utils import translate_sentence, save_checkopoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from elasticsearch import helpers, Elasticsearch
from Indices_Manager import IndexManager
import torch.optim as optim
import torch.nn as nn
import random
import torch


__device__ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

__DEBUG__ = True


def prep_train_data(es):
    """
    :param es: Elasticsearch client manager.
    :return: List of tuples of embeddings connecting mentions to entities [([float], [float]), (), ...]
    """

    pairs_generated = 0

    training_pairs = []
    response = helpers.scan(es.client, index="entity_mentions", query={"query": {"match_all": {}}})
    for doc in response:
        ent_lookup = es.client.search(index="entity_embeddings",
                                      query={"match": {
                                          "entity": doc['_source']['entity'][doc['_source']['entity'].rfind('/'):]
                                      }})
        for hit in ent_lookup['hits']['hits']:
            if hit['_source']['entity'] == doc['_source']['entity']:
                entity_embedding = hit["_source"]["embeddings"]

                for mention in doc['_source']['mentions']:
                    mention_emb_lookup = es.client.search(index="mention_embeddings",
                                                          query={"match": {
                                                              "mention": mention
                                                          }})
                    for m_hit in mention_emb_lookup['hits']['hits']:
                        if m_hit['_source']['mention'] == mention.lower():
                            training_pairs.append((m_hit['_source']['embeddings'], entity_embedding))
                            pairs_generated += 1
        if pairs_generated == 2 and __DEBUG__:
            return training_pairs

        if pairs_generated % 20 == 0:
            print(f"{(pairs_generated*100)/15909892}% Done")

    return training_pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        # self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, encoder_input):
        # input_shape = (seq_length, batch_size, embedding_size)
        outputs, (hidden, cell) = self.rnn(encoder_input.unsqueeze(0))
        return hidden, cell


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, decoder_input, hidden, cell):
        decoder_input = decoder_input.unsqueeze(0)
        outputs, (hidden, cell) = self.rnn(decoder_input, (hidden, cell))

        # Fully connected layer
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force=0.5):
        batch_size = source.size()[0]
        target_length = target.shape[0]
        target_vocab_size = 200

        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(__device__)
        hidden, cell = self.encoder(source)

        start_token = target[0]
        print(type(start_token))

        for t in range(1, target_length):
            output, hidden, cell = self.decoder(start_token, hidden, cell)

            outputs[t] = output
            best_guess = outputs.argmax(1)

            start_token = target[t] if random.random() < teacher_force else best_guess

        return outputs


def train_model(training_pairs, epochs, learning_rate):

    # Model hyper parameters
    input_size_encoder = 768
    input_size_decoder = 768
    output_size = 200

    hidden_size = 1024
    num_layers = 2
    encoder_dropout = 0.25
    decoder_dropout = 0.25

    writer = SummaryWriter(f"runs/Loss_plot")
    step = 0

    encoder_net = EncoderRNN(input_size_encoder, hidden_size,
                             num_layers, encoder_dropout).to(__device__)
    decoder_net = DecoderRNN(input_size_decoder, hidden_size, output_size,
                             num_layers, decoder_dropout).to(__device__)

    model = Seq2Seq(encoder_net, decoder_net).to(__device__)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        for pair in training_pairs:
            inp_data = torch.FloatTensor(pair[0]).unsqueeze(0)
            target = torch.FloatTensor(pair[1]).unsqueeze(0)

            output = model(inp_data, target)

            output = output.reshape(-1, output.shape[2])
            target = target.reshape(1, -1)

            print(output.size())
            print(target.size())

            optimizer.zero_grad()
            loss = criterion(output, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            writer.add_scalar("Training Loss", loss, global_step=step)
            step += 1

        print(f"Epoch [{epoch} / {epochs}], Loss = {loss}")





if __name__ == "__main__":
    # print("Starting Elasticsearch instance... ")
    # es_instance_manager = IndexManager(host_id="https://datasets-26efe5.es.us-central1.gcp.cloud.es.io:9243",
    #                                    key="CisJmi0r6jd7oIWqPFdB21Av")
    #
    # print("Prepping the training pairs")
    # training_data = prep_train_data(es_instance_manager)

    training_data = [([-2.52357173, -0.368530184, -1.81619132, -0.666339576, -2.27761626, 1.28451264, 1.49061739, -1.10915959, -1.76622593, 0.264122277, -0.0907326341, -0.140357226, -1.53932536, 2.12445521, -2.55737448, 4.38264894, 0.038807869, 1.36392844, 1.65279281, 1.24414098, -0.307364136, -1.12545049, -0.90682584, 3.1694634, -1.5057503, 2.70722365, 0.498841375, 1.71250761, -1.2425456, -1.34944952, -3.98310542, 2.95611286, 0.156075239, 1.87477875, 2.4898808, -0.307958931, 1.67687643, -0.258833945, 4.00469255, -2.88110805, -0.819428504, -0.230111599, 3.92360568, 2.10818291, -0.382993311, 0.321807534, -1.44558156, -2.69496465, 2.4708581, -0.0907083526, -2.57416797, -3.10336781, 2.65823388, -1.86458385, -0.458664656, 1.87774718, 2.68934941, -3.25637722, 2.45871139, -1.63805115, 2.22701192, 3.1502533, 1.37154806, -2.63121271, -1.68583548, 0.398412347, 3.63361859, 1.23351753, -1.46378326, -0.491605908, -1.8397733, 1.24768496, 0.29018569, 4.28805733, 3.54819679, -1.86241019, -3.68398881, 3.8096931, 0.189354658, 1.59826052, -0.581802905, -2.61504936, -0.217218205, -1.4544462, 5.35937691, 2.82822895, 2.07683039, -1.31161582, 1.96400321, 1.43055952, -0.588763535, -0.556081772, 0.702571869, -1.27567804, -1.00096953, -2.34356236, 0.746127784, -2.2453239, -2.20874882, 1.57096386, 0.733598709, -1.34451807, 4.18674707, 1.06543565, -1.77169073, -2.98734307, -0.41987291, -1.94629872, -2.31841397, 0.768786967, 0.503681362, 0.0866477862, -0.296534866, -3.70948148, -0.866326571, -3.66283226, -0.628684342, -3.9733398, -0.0142391324, 1.97663295, 1.74405611, 1.30473506, 0.166045308, 2.13105702, 0.348247558, 2.13123965, -1.52327299, 0.249773428, 0.694060028, 0.909542561, 2.82624888, -0.0779426619, 0.359205335, 0.70301348, 1.97016716, 0.894209564, 0.315482169, -0.475121528, 0.813156664, -4.09402037, -4.67182302, -2.13724017, -0.0750233307, -2.39551258, 2.06370139, 0.330714524, 1.01826441, -2.32566905, 0.18564342, 0.651533067, -1.92012215, 3.55194855, -1.24016988, 1.02379429, -0.398741484, -0.646248281, 0.452085376, -0.897801101, 0.277023315, -2.62959385, -1.92925835, 1.97068691, -2.28081822, -0.78849721, 1.34639502, -2.30681705, 2.43096638, -2.65308762, -1.81499755, -5.08773136, 0.123554386, -1.81277084, 1.7298125, -1.93527496, -1.90204668, 0.640571952, 0.0953653455, -1.44156682, -0.360385746, 1.54342425, -0.0571998358, -1.5126549, 1.2270714, 1.86446095, 2.2383461, 1.77318895, -0.500837207, 1.1017164, 0.482566684, 0.138490126, -0.325169235, 4.38519907, -0.255934089, -1.4900732, 2.70029092, 3.2867887, 1.34929466, -2.05401993, 0.917951643, 2.20812917, -2.04152822, -2.05657458, 2.79136205, -1.19769537, -0.735882998, 1.76675856, 1.89962244, -0.599709809, -4.54254341, -0.178920701, -1.84897101, 1.32228768, -0.922098577, 2.1910975, 0.367996216, 1.23410439, -1.40880001, -1.04657555, 3.4193027, -0.142669097, 1.52594817, 4.0471468, -5.04891586, 1.47237873, 1.40010083, 4.81849241, -0.981861889, -2.02732348, -0.790908158, 1.80161202, 2.77129817, -0.0629021302, 2.88706279, -0.958576143, -0.33303687, 1.2326926, 2.77230239, -1.02687049, 0.636004567, -1.71335065, -1.51011753, 2.823735, 1.32163036, 1.71267211, -0.934798062, -1.16557181, -1.30343306, 0.0812823772, -3.32256031, -1.91006088, -0.50574702, 1.84414566, 0.572584152, 0.0443218946, 0.577026844, -3.17515969, -0.710839331, -1.60604763, -2.72948003, 0.998204231, 0.757194519, 0.588868558, -1.46303511, -1.56279123, -0.853466809, -0.0831250027, 0.0570707507, 1.89086187, 0.943149805, 2.2521832, -0.985866487, 3.25073433, -2.48717141, 3.80222726, -0.771420002, 4.97205877, -1.91150284, 1.84531927, 2.74173331, 0.745302975, 2.89959049, 1.33900452, 2.39585304, -2.31787491, -5.78404474, 0.567979991, -0.00145816803, 1.49200189, -1.73028183, 0.0304258261, -1.1310941, 1.45174634, -2.52627754, -0.391332895, 2.52506757, -1.55216312, -0.515022755, 0.385828525, 0.907496631, -4.32137442, -3.45591426, -2.31233811, 4.14814091, 2.01872396, 4.25078392, 1.77724838, -0.267445832, -3.28133798, -22.0862732, -0.328872681, -1.43903911, -2.5680027, 3.67323232, -0.322918773, -1.71924174, 1.65913641, 0.498459011, -0.277663797, 0.701210976, -1.12252533, 1.20170975, -0.834887087, -2.27409577, 0.40472427, 2.80853009, 3.03077388, -0.497829646, 0.196587458, -1.97641981, -5.29484177, 0.923378408, 0.454299361, 1.50490999, -0.51064992, -4.50971651, 0.875282764, 0.939963579, -1.09388101, -3.26729012, -3.0660603, -0.736394167, 2.56910014, -0.937030315, 0.362762302, 0.480094165, -1.07216799, 0.168233156, -3.67140555, 0.115268029, -7.04452181, -0.536266506, 0.0382424109, -0.246103272, 1.82947886, 1.38648844, 0.850354493, -0.65364027, 4.20462227, 1.23118174, -0.575484514, -4.16570997, -2.52949429, -0.628400683, 1.10784435, 1.76538885, 2.2190764, 2.7670517, -2.45571041, -3.06291318, -1.00504673, -3.59909558, 0.875818431, -1.33773124, 1.5590564, -0.982445717, 1.33680284, 0.285648406, 2.2215085, -5.5875535, -1.59969509, 0.78935051, -4.79763269, -1.26551497, 0.766614914, 1.99639845, 1.54953325, -3.48019624, 0.159611464, -1.79474652, -2.45104671, -0.15479292, -3.45175552, -1.87757266, -1.19301546, 0.304857105, -2.52077079, -2.01745725, 0.685180426, 2.62018418, -0.557698727, -0.694051564, -2.78655696, 0.87769109, -0.182764769, 0.927519977, 1.92779779, -2.61687493, -1.14625764, 1.89439678, -2.86137486, -0.814163923, -3.99581122, 4.31217909, 1.4753437, 2.88657284, -0.43108806, -2.77988601, 0.0049069724, 1.05145586, 0.672762692, -0.567999303, -1.90190113, 0.925080121, -2.8852551, 0.895292103, 0.757014751, -0.322900265, -2.0388248, -3.58594346, -0.0184863713, 1.13046312, -6.2591095, -2.6800878, 1.64081347, -2.44342637, -1.21857917, -0.940454781, 1.75985563, 0.195878521, -0.124611259, 1.70474434, -1.06292474, -1.27310979, 0.583522379, -1.25139153, -2.90762711, -2.64492345, 2.07718372, 3.79533768, 0.0215071347, -0.382699728, 2.80274415, -1.61019528, 1.00799859, 0.547234058, -1.58077681, 0.0843475759, -2.92328191, 0.227375343, -2.51113129, 1.64064372, -3.32268596, 1.16671145, 2.23808622, 1.92221355, -0.45851782, -2.48029399, -3.16025996, 1.57906973, -0.862397373, 2.82654572, 1.23374426, -2.11712193, -1.27839315, -0.44720462, 4.00238037, 2.79874802, 0.0707012787, -0.403759003, 1.61204243, 0.963968575, -0.800977945, 1.09878945, 2.79192758, 2.74637628, -0.438776165, -2.42102361, 4.34707785, -1.60519111, 0.852243721, 1.14859545, 0.246869013, -1.02444065, -1.85099709, 2.49102759, -1.14501226, -1.07692301, -4.11480761, -2.08937693, 1.00176919, 1.82237208, -2.34670925, 0.501122653, -2.93457055, 1.34558117, -0.0243105087, -1.50470877, -1.05448258, -0.446784645, 0.560759723, -1.41034603, 0.487950712, -0.7762869, -1.34512901, -3.08518124, -0.137913302, 3.98795891, -1.68520308, -2.14596629, -3.2026577, -4.11874104, -1.37470818, 1.83820879, -0.661899984, 1.83709061, -1.49463975, -2.74260736, 0.286723047, -0.322217554, 2.30122662, 1.36388397, -0.792917907, -2.88037276, 1.88624823, 0.892009556, -2.19817162, -2.45585656, 1.48136294, -0.11302463, -7.48183584, 0.607837498, 2.08267522, -0.992379487, -0.28095302, -2.67717838, -2.94073224, 1.80873311, 1.4975394, -0.660497367, 1.85936642, 0.625028312, -1.19447887, 1.82777464, -0.269382328, -1.25033283, 0.6622172, 2.38183308, 0.229972005, -0.733869553, -0.313462526, -3.2562685, -1.89965975, 0.977434337, -1.7658143, -1.60170329, 1.07383144, 0.40409264, -2.28583455, -1.84959364, 2.40616131, -1.06886041, 0.453263849, -2.79789186, 0.407984495, 0.419685841, 0.719820738, -1.00140512, -2.57890511, 4.08163643, 0.145285383, 1.39328277, 0.681712449, -1.11250246, -0.715491474, -1.01472616, -2.06819606, 0.414399743, -1.85465515, -0.645975292, 0.64970535, -1.55230999, -2.86167645, 0.540084779, -3.21337628, 0.447307914, -0.543494761, -0.735616744, 4.56591415, 0.618029356, -1.90226519, -0.465191364, 1.16516423, -3.34552765, 2.0809319, -0.437054187, 2.63826203, -0.0866101608, 2.36925864, -0.641468585, -0.211501181, 1.47266865, 2.44628263, 1.02745783, 1.1839509, -1.04528904, 0.549275458, -1.55145788, 3.49447751, -1.83342266, -3.07846141, 2.23289275, 0.611028016, 2.04195333, -0.0683158264, -1.54637372, 0.679425955, 0.657066166, -2.94893432, -0.22233741, 0.574524105, 3.30703092, -1.49534237, 1.37940562, 1.25704324, -3.49873161, 1.83663714, -1.8939209, 2.96310043, -1.88359737, -0.788840473, 0.34907648, -0.752156019, -0.846038163, -0.0063839755, 0.941674292, -0.95970422, -4.70310545, 2.84819412, 1.3501755, -1.60494411, -2.8835299, -0.133660644, 4.46329832, -1.43258727, 1.22363317, 1.67736208, 2.44880199, 1.25071192, -0.42651248, -0.914870083, 4.03469133, 1.58897972, 1.70128584, 1.55385721, -2.20290089, -2.26898026, 2.56581569, 0.814875066, 0.482193708, 2.28832436, 3.54146266, -2.8154552, -2.04236221, 1.97384024, 0.443244785, 1.68217134, -0.945135415, -0.0435393639, -0.940317214, -0.868693888, 2.64680696, 0.703957617, 0.09766078, 0.548109591, -1.36364472, 1.01078713, 4.9131732, 0.761185646, -4.20430756, 0.718189895, -2.35813808, -0.708997071, 0.243727162, -1.6390872, 0.243068054, -1.66412365, -6.54760933, -1.45647776, -1.3295089, 0.654693961, -0.925080597, 1.42962468, -1.43961287, 1.13432491, -0.208416104, -0.423670858, 3.94113612, 1.05631793, 2.33948183, 0.377817631, -3.42970848, -1.80993092, 0.188704967, 0.849006474, -0.00913947821, 2.76490784, -0.935082138, 0.00942653418, 1.1092931, 1.42724931, -0.881542504, -0.542039573, 1.18689346, 0.90494585, 0.784506142, -1.27070284, -0.250101447, 1.98200381, -3.94057775, -3.60343146, 1.92815256, -1.87462342, 0.585013688, -0.736615241, 0.885402501, -2.20900559, -0.283642203, 0.410289645, 3.08937478, 2.76885343, 0.459945351, 1.46601498, 0.704703331, 0.984293282, -2.29856467, 4.38807058, -0.572711408, 0.441037565, 0.497865677, -0.136881754, 0.337647885, 1.65874231, 0.229162708, -2.08473754, -0.610459626, -1.31878376, -0.407589108, 2.05948853, -0.945390761, -0.593040049, 0.223069668, 2.30966306, -0.401324719, -0.833583593, -2.00586295, -1.77983534, -0.73785615, -0.0953198075],
                      [0.20582457, -0.5824308, 0.14748597, -0.13406289, -0.46566123, 0.09037858, 0.0294982, 0.27516693, -0.14780328, -0.3327535, -1.4177682, -0.17693773, -0.70291054, -0.25964797, -0.5188829, -0.38185516, -0.26793566, 0.38156044, -0.5147925, -0.20434591, -0.047124658, 0.26742816, 0.18137251, 0.064563155, -0.3915719, 0.69911224, -0.7717855, -0.40082076, 0.36835995, 0.42372763, 0.5994653, 0.43536666, -0.3549705, -0.47533557, -0.24883777, 1.2952436, -0.2935905, 0.88381124, 0.7949494, 0.566812, 0.17287904, -0.6028848, -0.29019064, 0.8343171, 0.40818733, -0.095292576, 0.7219279, -0.27455288, -0.16484337, 0.4637947, -0.15519586, -0.35316765, 0.84711987, 0.116662666, 0.1516261, 0.46275777, 0.1067242, 0.35808867, -0.51286703, -0.10422003, 0.009645012, 0.5291883, 0.44870704, -0.13720627, -0.42634383, 1.0045023, 0.6075491, -0.111152686, -0.40434277, -0.707473, -0.043868113, 0.09210869, -0.1346909, -0.21141548, 0.3443734, 0.45488822, 0.42760825, -0.4161114, 0.015500652, 0.655436, 0.07549601, 0.6722527, -0.099038996, -0.019197367, 0.5641636, -0.35208443, 0.23637778, 0.098898955, 1.2738506, 0.41349697, 0.3423321, -0.76399636, 0.39918515, 0.4492315, -0.81066585, 0.668043, 0.57131976, -0.68600196, 0.12067339, -0.5680841, -0.44676325, -0.0492109, 0.116190575, 0.11637612, 0.69742733, 0.023192441, 0.47423816, -0.03233211, -0.8851652, 0.37048057, -0.3417753, 0.09078903, 0.12509333, 0.6732805, 0.045557544, 0.112602584, -0.7472405, 0.33699092, -0.03664258, -0.22898263, 0.05702164, -0.08839991, 0.86041045, 0.43652418, -0.3786333, 0.9550647, -0.197225, 0.10065724, 0.09672015, 0.09121204, -0.4147704, -0.007099354, -0.22820468, -0.21723352, -0.47854537, 0.061619606, -0.4247818, -0.19332217, 0.30964282, -0.25139427, 0.41873613, -0.25640607, -0.28978246, 0.32720467, 0.23463233, 0.13574694, 0.32747, 0.6443684, -0.46775967, 0.14143707, 0.30886382, -0.22100651, -0.49826336, -0.06065204, 0.8153661, 0.56724215, -0.2088448, 1.0362042, -0.21299982, -0.7610544, 0.45299512, -0.140624, -0.42803285, 0.504548, -0.07945005, -0.83654124, -0.129784, -0.28336734, -0.31623307, -0.055603974, 0.40562603, 0.14595896, 0.32872075, 0.39829996, 0.47550052, 0.15928164, -0.28517616, 0.4585602, -0.053408705, -0.09185863, -0.35318184, -0.043246876, 0.5820972, -0.3891572, -0.76046425, -0.40384153, 0.09961278, 0.472011, 0.015241594, 0.102540374, 0.41994348, -0.14211659, 0.6698333, -0.1950242, 0.5119069, -0.28698105, -0.08277078, -0.41760355, -0.1554558, 0.9006636]),
                     ([-0.732950509, -1.15706646, 2.07947707, -1.62292457, 0.834765971, 4.49010181, 0.670993567, 2.8716104, 2.35406399, -2.93029213, -0.324526727, 0.240821287, 1.97591031, 1.28180337, -1.10317039, 2.65343165, -1.30720592, 4.00372124, 2.68495941, 0.034781456, 3.45522904, -3.52421665, -0.0361508131, 2.34188247, 3.08710957, 0.655168295, 0.19479984, 1.10958993, -1.7860043, 1.83993506, -2.5847702, 0.828972816, 2.62444735, 1.77787161, -3.66768098, -1.05980241, -0.531715989, -1.77393794, 0.593354344, -2.65205264, -1.5369401, -3.38846302, -0.577314854, 2.85697055, -1.21567583, -1.21657789, 2.75081325, 1.65356827, 1.10101473, 5.96521187, -2.52326441, -2.36564136, 1.89774597, 0.212090239, 0.668434381, 1.58069921, -0.0357403755, -2.78827262, 0.00620937347, -1.68632007, 2.36107278, 1.19744563, -1.82471824, 0.975943089, -0.536238432, 0.46996507, 2.26758838, -0.234007537, 1.26311195, -0.344711542, 0.8994205, 3.03059483, -1.14838481, -0.889968514, 0.781587481, 1.65296745, 0.0518656448, 0.555259824, 1.31587338, 0.460956722, -0.77445507, -0.640884876, 2.45086813, -1.30654597, 2.29371786, 0.857822716, -0.0509095192, -1.68495727, 1.61895287, 0.120171636, -2.17958999, 2.07104111, -0.434452146, -1.22318554, -0.753828883, 0.824499607, -2.06637669, -2.1355381, 0.183189869, -1.28563547, -1.814484, 1.49517989, 1.19188595, -0.0724760592, -1.43860888, -1.15276933, 0.257287323, -1.36005783, -0.0551451445, 0.0256904662, 0.74281621, -2.91454172, -0.0035956651, -1.68939137, 3.35264421, -3.06297922, 1.41437936, 0.617614388, 0.0912835598, 0.841905236, 2.4428997, -0.163529098, 1.72556281, 0.172047168, 2.9057436, 0.88426137, -2.77697229, -0.234147251, 1.91788518, 1.68123364, 0.988700867, 1.42518806, -0.639704525, 4.2780056, -1.88444734, 6.25193596, -1.3869803, -1.80463076, -0.221326947, -2.00908065, -0.565157354, -3.26092887, -2.77062988, -1.47211933, -1.46788406, 1.38240111, 1.7013011, 0.243293583, -2.97432971, -1.17290163, 0.0383904576, -2.75335121, 0.254085213, 2.60803318, -1.52194643, 0.137437224, -1.10900724, 0.541326404, -0.962306082, -0.870242238, 1.75891101, -0.515805244, -2.10844183, 1.72166944, -0.577970564, -2.52005076, -2.63281655, -1.92956245, -0.364301294, 1.83512473, -0.686469078, -0.445795715, 0.734634578, -3.82462001, -2.00338793, 0.936023355, -1.0566771, -1.15167856, -0.365316123, 3.79190254, 0.435432494, -1.86243153, 1.40351748, 1.67400789, 1.67246246, 5.85529947, -2.64476013, -0.857415438, -0.700052738, 0.0150427222, 2.72197294, 1.64763618, 2.59395957, -0.965178251, 0.171324164, -0.35025233, 4.44529867, -1.04143274, -0.112667441, -2.94798326, -0.395284534, -0.339988559, 1.47397912, -0.749261022, 0.717619121, 0.239697635, 0.328589797, -3.24784112, -1.60530758, 0.67386657, -1.64861298, 1.00064921, -0.254792094, -2.97519875, -1.28049779, -1.40955806, -2.06485462, -2.87845182, 3.57014275, -2.03720617, -0.821134806, 2.64803815, 0.490765095, 1.26927602, 0.946347594, 4.34768486, -0.263416231, 0.213607311, 2.96102667, 0.756156921, -1.20678508, -0.892144561, 0.0599205494, -3.42292452, -1.96541107, 1.49653959, 2.59492183, -1.64313889, 0.954141796, -2.3683176, 0.26677224, 4.31581736, 0.182312965, -1.36554408, -0.130809069, -1.04902792, -0.306316018, 2.15946913, -1.0695138, -1.65147233, 1.60387564, 5.33970213, -0.817763746, 2.49480629, -1.36513209, 0.226657629, -0.25121513, -0.9912889, -0.33917284, -2.75471783, 1.72731197, 1.11386073, -0.44060564, 0.678904653, -2.67372417, -0.986488342, -1.49464321, 0.986944199, 1.05362201, -0.785885513, -0.725334764, 2.9448061, -0.615133405, -0.625965714, 2.40956163, 0.649035096, -0.165379971, 1.25772333, 0.309359699, -0.040768683, 0.707192659, 1.52204001, 0.675614119, 1.34027088, -1.24492931, -0.110070795, 3.59366775, -1.57575011, 1.99426723, -3.84009409, 0.990842938, -1.98388648, 2.14801145, -0.0489321314, 0.380648792, -0.65640676, 2.7582233, -0.617257774, -0.624600053, -1.88357222, -6.01371336, -0.906517684, 4.18050289, -1.19378614, 8.74190044, 2.97742987, 2.2843399, -0.256628633, -36.7732925, -1.14939713, -1.85618877, -2.1597774, 2.92893744, -1.354918, -2.64286089, 2.0550456, -0.71739924, -3.28343678, -1.61200309, 1.44934535, -0.546651959, 0.332691908, -0.589376867, 0.296934783, 1.86217403, -3.13775945, -2.75016332, 1.41839254, 1.66221368, -3.68786597, 2.1457181, -1.01478839, 2.15935707, -1.68783283, -3.25360584, 1.62060475, -0.161488354, 2.09373474, 2.66437268, -1.7918458, 0.0240461826, 1.04676962, 1.02541471, 2.81876564, 0.563157201, -1.23020744, -0.275960565, -4.56677389, -1.51241684, -6.64070129, -3.24378467, 1.57618761, 0.56693238, 1.26957214, 0.669217765, -1.24971855, -1.30022228, 2.32157207, 1.56774902, -1.04767108, -3.01837873, -1.99694753, 0.907235563, -0.0822812319, -0.0752068758, -1.54237545, 0.900091052, 0.32096982, -2.18454981, 2.26098847, 1.47861481, -0.129308879, 0.0433573127, 0.109812677, -1.46514475, -5.47719383, 0.396268249, 1.89643645, 0.149120539, 1.14865565, 0.459014058, -4.35484505, -0.82687819, 0.769648373, 0.909733593, 1.05371249, -0.957969725, 3.17989159, 1.77952838, -0.793322802, -0.011316061, 0.805486441, 3.80454302, -1.09233212, -2.08532667, 3.56394577, -0.178303897, -0.784203947, 0.493422687, -3.18707657, -1.37774134, -1.47024393, 0.609072804, -1.66422796, -0.386338353, 0.796232641, 1.87216401, 1.25691319, 0.622420728, 1.22820354, -0.361691326, -3.55840349, -3.16887522, -0.409814358, -0.494239509, 1.05049789, 0.326840997, -1.97959328, 4.3975091, -0.358718276, -0.302087784, -0.141874075, 1.51285219, 0.459763885, -0.445870399, 1.42357767, 1.41685081, -3.19220781, -2.0609467, -0.995638072, 0.062669456, -1.7552768, 0.853864968, -1.18514383, -0.881048679, -0.952105224, -1.16037393, 1.17973781, 1.68925428, -0.739751935, 0.148436666, -0.430092514, -0.503062725, -2.23083568, -1.80603886, -3.56960154, -3.26600099, 2.64612174, -0.519435048, -2.55192614, -0.696320772, 2.17142439, -0.541243911, -0.270417303, -1.3342756, -1.75783789, 0.340296388, 2.05241489, 0.495642245, -2.81060958, 0.509214342, -1.19927335, -1.71368265, 1.92183065, 0.211477518, -1.46026659, -0.835832834, 0.464587361, 0.00957348198, -0.158899158, 2.46627378, -3.6130476, -3.12964797, -0.941830575, -1.58837676, 1.88156033, -4.35841274, -2.62194586, 0.899102926, -1.08614206, -2.24660134, 1.67687249, 0.458426476, 1.45525002, 2.82337523, -0.620207906, -2.30323148, 2.94786549, 0.193618596, -1.17436481, -0.923325419, 1.71658874, -1.15069234, 0.764010251, -1.44343066, -0.719081283, -0.847165108, -0.845151782, -1.05763555, 0.102024376, -0.494903564, -1.85605395, 0.501601636, 0.470584571, 0.659318089, 0.242636502, 0.650081515, -3.09547853, 1.69568276, -1.13030696, -3.30058002, -0.709723711, 0.379925966, 0.406493664, 0.582626879, 4.39703941, 2.11639261, -0.907584131, -5.68923426, -1.46035337, -3.2832408, 1.65981603, 0.505137563, 0.671239614, 1.44296277, -2.00458097, -1.89584756, -1.74981809, -1.51822019, 0.677270353, -0.426963687, -2.78860378, 0.418521434, 4.92600441, 4.19205379, -2.36305976, -1.28249049, -1.28386223, -0.390745908, -10.1719952, 0.396450698, 0.818723679, 2.36177278, -0.988015175, 1.29593801, -3.44596648, 0.969959855, -1.49524868, 2.32526135, 1.7716465, -0.879667759, -0.415217161, 1.26519275, 0.3012923, -0.0914340019, 0.95597887, -0.883427143, 2.66884279, 3.37301421, 1.92096424, 0.233791888, 1.5530659, 5.12051105, -0.747003436, -2.86055756, -1.21739185, -0.234214336, -0.0400848985, 0.546459734, -1.22488642, 0.635593534, 4.640872, 0.27233997, 1.96460223, -0.44057852, -2.07769847, 1.00525033, 0.0699872449, 2.74266195, 1.21783352, 4.1842804, 1.0967052, -1.87686098, -1.22073221, -4.57578373, -0.0480907857, -2.62087584, -2.79614496, 0.812730312, -1.02018702, -3.61527443, -2.76020122, 0.436229467, -0.988571942, 1.73223782, -1.74100184, -1.14386725, -1.4479506, 4.56426287, -1.21124363, 2.71655655, 0.451024801, 0.127192318, -0.0210480094, 0.0557094216, -0.198943466, -1.07819867, 2.16599894, 1.40495694, -1.72220266, 0.161673099, -0.966388106, -1.45052373, 1.1736666, -0.0815339088, 1.02936256, 0.548857689, 2.24357462, -4.58027077, -1.07844603, 1.35260153, -0.733389318, 2.12045765, 3.20040226, -0.203243136, -0.97923398, -0.381706595, -1.20385742, -0.12298584, 0.0186461806, 1.92403829, -0.537253439, 1.52288663, -0.810110807, -1.98554015, -0.720378697, 0.954676747, -1.79595971, -0.129824877, -0.879641533, 0.710688114, -1.15037501, -0.601648092, -1.37865973, 2.34696245, 1.33334279, -0.311013013, -0.263317823, 2.04893875, 1.9337225, -4.60239935, -0.743205607, -1.93005145, 0.216342628, 0.305019796, -1.31314456, -0.0398693979, -1.06269979, -0.0392392874, -1.22022963, -1.95015025, -2.09617662, 1.47945082, -1.46914887, 0.148741856, 0.774578035, 0.100717664, 1.34663951, -0.106164023, 2.95628405, 1.91095424, -3.02718329, -2.14066458, -1.82018495, -1.32736421, 0.0324171185, 1.82991815, 3.27219224, -2.12922263, -1.88075972, 1.97072887, 0.0973724127, 0.244984418, 0.158660769, -0.145227432, 0.982678175, 5.24383879, -3.38594103, 0.0662821531, -0.906675458, 1.20659375, -0.43827492, 1.46682501, -0.860103369, 0.72988236, 1.11439967, -1.51886082, -2.51546788, 0.47073698, -0.449734181, -5.02642059, -4.16290188, -0.833782375, -2.2961092, -3.49968958, 0.145932913, 0.156794652, 0.289302111, 1.13694692, -1.35766959, -0.446148545, 2.09606051, 1.57434976, -1.70102537, 1.4665494, -0.23230803, 0.337732553, 2.5625329, -4.24605799, 0.349954039, 0.0272613391, 0.656548738, 2.33890295, 1.23999906, -0.0116580427, 3.59449244, -4.58648109, 4.54489946, -1.27015018, -1.72255647, -0.180941775, -0.790664434, 0.514893413, -5.17443943, -1.1226846, -0.0893118382, -0.545747161, 1.76002932, -0.355685592, 0.662708402, -1.63891757, -0.275543988, -1.58126974, -0.0222997069, -1.00497198, -1.23636746, 1.1010071, 2.06975603, -1.00748968, 2.0504005, -4.95056438, 2.41306734, 1.97570944, 1.75007486, 0.0816290602, -1.74754262, 0.632116556, -0.593472004, 0.979006886, 0.451722503, 2.04357266, 0.748414516, 0.522805572, 0.75610888, -4.61889935, 1.03811216, -1.74503994, -0.594717085],
                      [0.5202767, -0.20958127, 0.106137976, -0.15469603, -0.10849959, 0.1506043, -0.051348977, -0.3734842, -0.2714308, -0.20971832, 0.09229747, 0.10584749, -0.06260034, -0.5314694, 0.6122799, -0.024147173, 0.13081618, 0.31637207, -0.016445005, -0.14745317, -0.44652766, 0.21258691, -0.33645937, -0.2449636, 0.063413195, -0.16381116, -0.32915726, 0.035325363, 0.62368906, 0.07878009, -0.5847177, -0.14157283, -0.30093563, -0.26981953, -0.31842813, 0.07605673, -0.5571648, 0.57014865, -0.10021613, 0.3316412, 0.022090342, 0.008915602, -0.19780682, 0.22012731, 0.34990197, 0.10526588, 0.51813924, -0.282131, -0.03239686, 0.4097044, -0.34455875, -0.530844, 0.22250067, -0.1064567, 0.38990957, 0.042047795, 0.34357446, -0.18824401, 0.09812372, -0.38569683, 0.22504173, 0.15501006, 0.646641, -0.12884007, -0.31514755, 0.40478572, 0.26626095, 0.036511973, 0.32800904, 0.22728486, -0.1802104, 0.09280139, -0.33601472, -0.20508258, -0.046288934, 0.5424161, 0.10182188, 0.24005203, 0.011986246, 0.051633023, -0.033627596, -0.33762082, 0.045467276, -0.19380847, 0.5390358, 0.08557966, 0.037164148, 0.048344456, 0.12238101, -0.03312492, 0.052401066, -0.26729232, 0.07277547, 0.070556104, -0.33503297, -0.06543631, 0.3170099, -0.19229051, 0.21981819, -0.03547457, -0.287797, -0.12847644, 0.31886554, 0.019036507, -0.01304498, -0.055605553, -0.09534459, 0.11989468, 0.1206176, 0.10973205, 0.13795263, -0.40059346, 0.14147174, 0.06573718, -0.42913637, 0.077384815, -0.088728234, -0.22463536, -0.16431692, -0.16563837, -0.10436629, -0.7511966, 0.34177563, -0.32835704, 0.22947764, 0.2512826, -0.07384728, 0.16214728, -0.21777959, 0.31396613, 0.39355326, 0.0016094075, 0.085378736, 0.2819302, -0.40382552, -0.6037932, -0.0031798396, -0.38104448, 0.2075802, 0.09861931, -0.36582997, 0.12948282, -0.17080882, 0.17448516, 0.15167326, 0.007540661, 0.3846626, 0.4215354, 0.16489089, -0.5142468, 0.15221457, -0.4865484, 0.40147328, 0.23634858, -0.29270443, -0.022367936, -0.11342842, -0.4501392, -0.03651588, -0.1005004, -0.22464229, -0.26104313, -0.26405424, 0.25948924, -0.8273444, -0.7198995, -0.38177118, -0.16999283, -0.27039462, 0.3890674, -0.18164225, 0.32916966, 0.18449275, 0.019949723, 0.38896665, -0.44106457, -0.2704219, 0.48504758, 0.15374696, -0.35343134, -0.06358488, 0.51626205, -0.008595321, -0.39171955, -0.13934033, -0.21801162, -0.4597055, 0.28291607, -0.46433344, 0.11334612, 0.5991669, 0.024436563, 0.26715818, 0.1826301, 0.24627538, -0.12221275, -0.35880944, 0.29469535, 0.2913955, -0.14360091])]

    print("Starting the model's training")
    train_model(training_data)
