import os


class HumanJudgementDataset:
    """#TODO
    """

    DATA_MAP = {
        'dailydialog_EVAL': ['transformer_ranker', 'transformer_generator'],
        'empatheticdialogues': ['transformer_ranker', 'transformer_generator'],
        'convai2':['transformer_ranker', 'transformer_generator',
                   'bert_ranker', 'dialogGPT'],
    }
    MIN_SCORE = 1.
    MAX_SCORE = 5.

    def __init__(self,
                 eval_data_dir_path,
                 dataset_name,
                 dialog_model_name=None,
                 auto_score_file_dict=None):
        self.eval_data_dir_path = eval_data_dir_path
        self.dataset_name = dataset_name
        self.dialog_model_name = dialog_model_name
        self.auto_score_file_dict = auto_score_file_dict
        if dialog_model_name:
            self.data_list = self._load_specified_data()
        else:
            self.data_list = self._load_mix_data()

    def _load_specified_data(self):
        data_dir_path = os.path.join(self.eval_data_dir_path,
                                     self.dataset_name,
                                     self.dialog_model_name)
        with open(os.path.join(data_dir_path, 'human_ctx.txt'), 'r') as f:
            context_list = [
                ctx.strip().split('|||') for ctx in f.readlines()
            ]
        with open(os.path.join(data_dir_path, 'human_hyp.txt'), 'r') as f:
            hyp_response_list = [
                hyp.strip() for hyp in f.readlines()
            ]
        with open(os.path.join(data_dir_path, 'human_ref.txt'), 'r') as f:
            ref_response_list = [
                ref.strip() for ref in f.readlines()
            ]
        with open(os.path.join(data_dir_path, 'human_score.txt'), 'r') as f:
            human_score_list = [
                float(score.strip()) for score in f.readlines()
            ]

        data_list = [
            {
                'context': context,
                'hyp_response': hyp_response,
                'ref_response': ref_response,
                'human_score': round(human_score, 2),
            }
            for context, hyp_response, ref_response, human_score in \
                zip(context_list, hyp_response_list,
                    ref_response_list, human_score_list)
        ]

        if self.auto_score_file_dict:
            for metric_name, file_name in self.auto_score_file_dict.items():
                with open(os.path.join(data_dir_path, file_name), 'r') as f:
                    for idx, score_str in enumerate(f.readlines()):
                        score = round(float(score_str.strip()) * 4 + 1, 2)
                        data_list[idx][metric_name] = score

        return data_list

    def _load_mix_data(self):
        data_list = []
        dialog_model_names = self.DATA_MAP[self.dataset_name]
        for dialog_model_name in dialog_model_names:
            self.dialog_model_name = dialog_model_name
            data_list.extend(self._load_specified_data())
        return data_list

    def categorize_data(self, num_bins=4):
        interval = (self.MAX_SCORE - self.MIN_SCORE) / num_bins
        categorized_data_dict = {}
        for i in range(num_bins):
            lower_bound = self.MIN_SCORE + i * interval
            upper_bound = lower_bound + interval
            selected_data = [data for data in self.data_list
                             if data['human_score'] >= lower_bound and \
                                data['human_score'] < upper_bound]
            key = 'score ∈ [{}, {})'.format(lower_bound, upper_bound)
            categorized_data_dict[key] = selected_data
        return categorized_data_dict

    def get_human_scores(self):
        human_scores = [data['human_score'] for data in self.data_list]
        return human_scores

    def __iter__(self):
        return self.data_list.__iter__()

    def __len__(self):
        return len(self.data_list)
