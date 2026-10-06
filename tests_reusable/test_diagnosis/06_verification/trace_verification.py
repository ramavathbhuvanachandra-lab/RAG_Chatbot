from tests_reusable.test_diagnosis.run_diagnosis import run_case
from tests_reusable.test_diagnosis.cases.benchmark import get_case
import argparse,json
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--case'); a=p.parse_args(); print(json.dumps(run_case(get_case(a.case),save_state=True),indent=2,ensure_ascii=False,default=str))
