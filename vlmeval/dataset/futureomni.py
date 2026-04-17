import re
import os.path as osp
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'

_PROMPT_VL = (
    'These are the frames of a video. '
    'Select the best answer to the following multiple-choice question based on the video. '
    'Respond with only the letter (A, B, C, D, E, F) of the correct option.'
)

_OPTION_LETTERS = list('ABCDEF')


class FutureOmni(VideoBaseDataset):
    """FutureOmni video-QA benchmark (up to 6-option MCQ, video-only path).

    Data layout expected under LMUDataRoot()/FutureOmni/:
        futureomni_test.json   – annotations
        videos/{qid}.mp4       – video clips

    JSON item fields:
        qid (int), source (str), question (str), options (list[str]),
        video (str, original name), seconds (float, clip end time), answer (str letter A-F).
    """

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='FutureOmni', nframe=32, fps=-1, adaptive=False):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps, adaptive=adaptive)

    # ------------------------------------------------------------------ #
    #  Registration
    # ------------------------------------------------------------------ #
    @classmethod
    def supported_datasets(cls):
        return ['FutureOmni']

    # ------------------------------------------------------------------ #
    #  Data preparation: build TSV from source JSON
    # ------------------------------------------------------------------ #
    # Hardcoded server path; falls back to LMUDataRoot()/FutureOmni if not present
    _DATA_ROOT = '/m2v_intern/xuboshen/zgw/data/FutureOmni'

    def prepare_dataset(self, dataset_name='FutureOmni'):
        if osp.exists(self._DATA_ROOT):
            data_root = self._DATA_ROOT
        else:
            lmu_root = LMUDataRoot()
            data_root = osp.join(lmu_root, 'FutureOmni')
        video_root = osp.join(data_root, 'videos')

        json_file = osp.join(data_root, 'futureomni_test.json')
        tsv_file = osp.join(data_root, f'{dataset_name}.tsv')

        if not osp.exists(tsv_file):
            assert osp.exists(json_file), (
                f'FutureOmni annotation not found: {json_file}\n'
                f'Please place futureomni_test.json under {data_root}/'
            )
            with open(json_file, 'r') as f:
                items = json.load(f)

            rows = []
            for idx, item in enumerate(items):
                qid = item['qid']
                options = item.get('options', [])
                # options already contain letter prefixes (e.g. "A. xxx"), join directly
                option_str = '\n'.join(options[:len(_OPTION_LETTERS)])
                question_full = item['question'].rstrip() + '\n' + option_str

                # GT answer: normalise to single uppercase letter
                raw_ans = item.get('answer', item.get('gt_answer', item.get('ans', '')))
                answer = str(raw_ans).strip().upper()
                if answer and answer[0] in _OPTION_LETTERS:
                    answer = answer[0]

                rows.append({
                    'index': idx,
                    'video': str(qid),       # e.g. "0", "1000"
                    'question': question_full,
                    'answer': answer,
                    'seconds': item.get('seconds', None),
                    'source': item.get('source', ''),
                    'qid': qid,
                })

            df = pd.DataFrame(rows)
            os.makedirs(data_root, exist_ok=True)
            df.to_csv(tsv_file, sep='\t', index=False)

        return dict(data_file=tsv_file, root=video_root)

    # ------------------------------------------------------------------ #
    #  Prompt building
    # ------------------------------------------------------------------ #
    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            line = self.data.iloc[line]

        qid = line.get('qid', line['video'])
        video_path = osp.join(self.data_root, f'{qid}.mp4')

        message = [dict(type='text', value=_PROMPT_VL)]

        if video_llm:
            message.append(self.make_video_struct(video_path))
        else:
            frames = self.save_video_frames(line['video'])
            for frame in frames:
                message.append(dict(type='image', value=frame))

        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # ------------------------------------------------------------------ #
    #  Evaluation
    # ------------------------------------------------------------------ #
    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from vlmeval.utils.matching_util import extract_answer_from_cot, parse_options_from_question
        from vlmeval.dataset.utils.multiple_choice import extract_answer_from_item

        data = load(eval_file)
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            # GPT judge setup (default exact_matching — no API calls)
            model = judge_kwargs.setdefault('model', 'exact_matching')
            if model == 'exact_matching':
                model = None
            else:
                from vlmeval.dataset.utils.judge_util import build_judge
                model = build_judge(**judge_kwargs)
                if not model.working():
                    model = None

            unparsed_count = 0
            for idx, row in data.iterrows():
                raw_pred = row.get('prediction', '')
                # Treat pandas NaN / None / literal "nan" as missing
                if pd.isna(raw_pred):
                    data.loc[idx, 'score'] = -1
                    data.loc[idx, 'extracted_answer'] = ''
                    continue
                pred = str(raw_pred).strip()
                ans = str(row.get('answer', '')).strip().upper()
                if not ans or not pred or pred.lower() == 'nan':
                    data.loc[idx, 'score'] = -1
                    data.loc[idx, 'extracted_answer'] = ''
                    continue
                pred_letter = extract_answer_from_cot(pred, valid_options='ABCDEF')
                if not pred_letter:
                    # Fallback: can_infer / GPT judge via extract_answer_from_item
                    options = parse_options_from_question(str(row.get('question', '')))
                    if options:
                        item = {**row.to_dict(), **options}
                        result = extract_answer_from_item(model, item, 'FutureOmni')
                        pred_letter = result['opt'] if result['opt'] != 'Z' else ''
                if not pred_letter:
                    unparsed_count += 1
                data.loc[idx, 'extracted_answer'] = pred_letter
                data.loc[idx, 'score'] = int(pred_letter == ans)
            if unparsed_count > 0:
                print(f'[FutureOmni] WARNING: Failed to parse answer for {unparsed_count}/{len(data)} samples')
            dump(data, score_file)
            # Also save JSONL for easy server-side viewing
            jsonl_file = score_file.rsplit('.', 1)[0] + '.jsonl'
            dump(data, jsonl_file)
        else:
            data = load(score_file)

        # Build result DataFrame: per-source + overall
        result_board = {}
        for _, row in data.iterrows():
            src = str(row.get('source', 'all'))
            if src not in result_board:
                result_board[src] = [0, 0]
            result_board[src][1] += 1
            if row.get('score', -1) == 1:
                result_board[src][0] += 1

        correct = sum(v[0] for v in result_board.values())
        total = sum(v[1] for v in result_board.values())
        result_board['overall'] = [correct, total]

        rows = []
        for key in sorted(result_board.keys()):
            c, t = result_board[key]
            rows.append({
                'category': key,
                'success': c,
                'overall': t,
                'accuracy': round(c / t * 100, 2) if t > 0 else 0.0,
            })
        result_df = pd.DataFrame(rows)

        acc_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(result_df, acc_file)

        accuracy_str = f'{correct}/{total} = {correct / total:.2%}' if total > 0 else 'no samples'
        print(f'FutureOmni Accuracy: {accuracy_str}')
        for _, r in result_df.iterrows():
            if r['category'] != 'overall':
                print(f'  [{r["category"]}] {r["success"]}/{r["overall"]}')
        return result_df
