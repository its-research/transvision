"""Data-free Linux CPU calibration runtime candidate probe; never paper evidence."""
import json,hashlib,sys,platform,os
from pathlib import Path
from importlib.metadata import version,distribution
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
SOURCE_FUNCTION_BINDINGS = [{'source': 'transvision/models/event_track_v2x/prediction_features.py', 'function': 'wrap_angle', 'source_file_sha256': '770b35f013d8c0d0c0bf1be4a2d814240b64a5f853970709281899c2445cd278', 'function_ast_sha256': '7e2b0580580b07911431c19b707044dabde8f7f790a198c11548aeae39e8cc2f'}, {'source': 'transvision/models/event_track_v2x/prediction_features.py', 'function': 'fit_score', 'source_file_sha256': '770b35f013d8c0d0c0bf1be4a2d814240b64a5f853970709281899c2445cd278', 'function_ast_sha256': 'ed921e782d7a18f7821879c8ad6e5e94596547fd3f3dedc74fa318f8970560a8'}, {'source': 'transvision/models/event_track_v2x/prediction_features.py', 'function': 'fit_covariance', 'source_file_sha256': '770b35f013d8c0d0c0bf1be4a2d814240b64a5f853970709281899c2445cd278', 'function_ast_sha256': '1971c432aac716afa1c1415304b592609efd048e73cefccc2c12e92cbf101693'}, {'source': 'tools/event_track_v2x/verify_spd_canonical_calibration_parameters_candidate.py', 'function': 'need', 'source_file_sha256': '35099319fa3a33e7a1b21853b1f7d959f203331766bd8922c809d754ffb85180', 'function_ast_sha256': 'a06f923ac6223241a70d80105c35b555016a9dccf5d416b0bf2596566ce088dd'}, {'source': 'tools/event_track_v2x/verify_spd_canonical_calibration_parameters_candidate.py', 'function': 'close', 'source_file_sha256': '35099319fa3a33e7a1b21853b1f7d959f203331766bd8922c809d754ffb85180', 'function_ast_sha256': 'b5d0b76f8fcd77231c592de5a3b658fea091dbeb8ac8528c568f1d96ff835a5f'}, {'source': 'tools/event_track_v2x/verify_spd_canonical_calibration_parameters_candidate.py', 'function': 'score_oracle', 'source_file_sha256': '35099319fa3a33e7a1b21853b1f7d959f203331766bd8922c809d754ffb85180', 'function_ast_sha256': 'ad9cac9a9c05ae7ff71869a9184bf9ff74388f53001ded0ea1b53f6afd3f58f6'}, {'source': 'tools/event_track_v2x/verify_spd_canonical_calibration_parameters_candidate.py', 'function': 'covariance_oracle', 'source_file_sha256': '35099319fa3a33e7a1b21853b1f7d959f203331766bd8922c809d754ffb85180', 'function_ast_sha256': '034fd9e663fb460adcc2cbd1f6ee689664947bbac5d284dcbec483432b27b24f'}]
def wrap_angle(value):
    return (np.asarray(value) + np.pi) % (2 * np.pi) - np.pi

def fit_score(scores, targets, config):
    from scipy.optimize import minimize
    from scipy.special import expit
    scores, targets = np.asarray(scores, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if not len(scores) or scores.shape != targets.shape or np.any((targets != 0) & (targets != 1)):
        raise ValueError("invalid calibration examples")
    eps = config["logit_clip"]
    p = np.clip(scores, eps, 1 - eps)
    x = np.log(p / (1 - p))
    prior = (targets.sum() + 1) / (len(targets) + 2)
    def objective(theta):
        z = theta[0] * x + theta[1]
        residual = expit(z) - targets
        loss = np.mean(np.logaddexp(0, z) - targets * z) + config["l2"] * theta[0]**2 / 2
        grad = np.array([np.mean(residual * x) + config["l2"] * theta[0], residual.mean()])
        return loss, grad
    result = minimize(objective, [0.0, np.log(prior / (1 - prior))], jac=True,
                      method="L-BFGS-B", bounds=[(0.0, 20.0), (-30.0, 30.0)],
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("score calibration did not converge: " + str(result.message))
    calibrated = expit(result.x[0] * x + result.x[1])
    return {"slope": float(result.x[0]), "intercept": float(result.x[1]),
            "examples": len(scores), "positives": int(targets.sum()),
            "in_sample_brier_raw": float(np.mean((scores - targets)**2)),
            "in_sample_brier_calibrated": float(np.mean((calibrated - targets)**2)),
            "optimizer_converged": True, "logit_clip": eps}

def fit_covariance(residuals, config):
    r = np.asarray(residuals, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] != 9 or not len(r) or not np.isfinite(r).all():
        raise ValueError("invalid covariance residuals")
    r = r.copy()
    r[:, 6] = wrap_angle(r[:, 6])
    second = r.T @ r / len(r)
    shrink = config["shrinkage"]
    cov = (1 - shrink) * second + shrink * np.diag(np.diag(second)) + np.diag(np.square(config["floor_std"]))
    np.linalg.cholesky(cov)
    return {"matrix": cov.tolist(), "samples": len(r), "mean_residual_not_corrected": r.mean(0).tolist(),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(cov).min()),
            "interpretation": "conditional_on_2m_true_positive; includes_bias_second_moment; not_false_positive_uncertainty"}

def need(ok, message):
    if not ok:
        raise ValueError(message)

def close(actual, expected, name):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    need(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
         and np.allclose(a, b, rtol=1e-10, atol=1e-12), 'recomputed parameter differs: '+name)

def score_oracle(scores, targets):
    x = np.log(np.clip(scores, 1e-6, 1-1e-6)/(1-np.clip(scores, 1e-6, 1-1e-6)))
    y = targets.astype(float)
    def objective(theta):
        z = theta[0]*x + theta[1]
        residual = expit(z)-y
        return (float(np.mean(np.logaddexp(0, z)-y*z)+.0001*theta[0]**2/2),
                np.array([np.mean(residual*x)+.0001*theta[0], np.mean(residual)]))
    prior = (y.sum()+1)/(len(y)+2)
    optimum = minimize(objective, [0., np.log(prior/(1-prior))], jac=True,
                       method='L-BFGS-B', bounds=[(0., 20.), (-30., 30.)],
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
    need(optimum.success and np.isfinite(optimum.x).all(), 'independent score fit did not converge')
    calibrated = expit(optimum.x[0]*x+optimum.x[1])
    return {'slope': float(optimum.x[0]), 'intercept': float(optimum.x[1]),
            'examples': len(y), 'positives': int(y.sum()), 'logit_clip': 1e-6,
            'in_sample_brier_raw': float(np.mean((scores-y)**2)),
            'in_sample_brier_calibrated': float(np.mean((calibrated-y)**2)),
            'optimizer_converged': True}

def covariance_oracle(residuals):
    values = residuals.copy()
    values[:, 6] = (values[:, 6]+np.pi) % (2*np.pi)-np.pi
    second = values.T@values/len(values)
    matrix = .9*second+.1*np.diag(np.diag(second))+np.diag(np.square([.05]*6+[.01, .1, .1]))
    np.linalg.cholesky(matrix)
    return matrix, values.mean(0), float(np.linalg.eigvalsh(matrix).min())

def numerical_probe():
 rng=np.random.default_rng(1337)
 result=[]
 for offset in range(3):
  scores=np.linspace(0,1,128);targets=((np.arange(128)+offset)%3!=0).astype(float)
  fitted=fit_score(scores,targets,{'logit_clip':1e-6,'l2':.0001});oracle=score_oracle(scores,targets)
  for key in ('slope','intercept','in_sample_brier_raw','in_sample_brier_calibrated'):close(fitted[key],oracle[key],key)
  residuals=rng.normal(size=(64,9));residuals[:,6]+=8*np.pi
  covariance=fit_covariance(residuals,{'shrinkage':.1,'floor_std':[.05]*6+[.01,.1,.1]});matrix,mean,minimum=covariance_oracle(residuals)
  close(covariance['matrix'],matrix,'covariance');close(covariance['mean_residual_not_corrected'],mean,'mean');close(covariance['minimum_eigenvalue'],minimum,'minimum')
  result.append({'case':offset,'score':fitted,'covariance':covariance,'independent_numeric_agreement':True})
 return result

def main():
 if '--local-check' in sys.argv:
  print(json.dumps({'cases':numerical_probe(),'data_read':False,'linux_runtime_admitted':False}));return
 from clearml import Task
 task=Task.init(project_name='Thesis/Recover-Before-Fuse/Training',task_name='SPD canonical calibration Linux L40S CPU data-free runtime candidate',reuse_last_task_id=False,auto_connect_frameworks=False,auto_connect_arg_parser=False);task.output_uri=True
 expected={'numpy':'1.26.4','scipy':'1.14.1','clearml':'2.1.5'}
 observed={n:version(n) for n in expected}
 observation={'expected_versions':expected,'observed_versions':observed,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'dataset_read':False,'actual_calibration_fit_verified':False,'paper_eligible':False}
 observed_path=Path('/tmp/canonical-calibration-runtime-observation.json');observed_path.write_text(json.dumps(observation,indent=2)+'\n')
 need(task.upload_artifact('runtime-observation',artifact_object=observed_path,wait_on_upload=True),'runtime observation upload failed')
 print('CALIBRATION_RUNTIME_IDENTITY '+json.dumps(observation),flush=True)
 need(observed==expected,'candidate package versions differ')
 need(platform.system()=='Linux' and sys.version_info[:2]==(3,12),'Linux Python3.12 candidate required')
 need(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only probe required')
 thread_keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
 need(all(os.environ.get(k)=='1' for k in thread_keys),'single-thread runtime differs')
 packages={}
 for n in expected:
  d=distribution(n);records=[x for x in d.files if str(x).endswith('.dist-info/RECORD')];need(len(records)==1,'distribution RECORD absent')
  packages[n]={'version':version(n),'record_sha256':hashlib.sha256(d.locate_file(records[0]).read_bytes()).hexdigest()}
 result={'kind':'canonical_calibration_linux126_cpu_runtime_candidate_probe','platform':platform.platform(),'python':sys.version,'packages':packages,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_function_bindings':SOURCE_FUNCTION_BINDINGS,'cases':numerical_probe(),'threads':{k:os.environ[k] for k in thread_keys},'dataset_read':False,'parameter_training_on_experiment_data':False,'actual_calibration_fit_verified':False,'formal_v2_ready':False,'paper_eligible':False}
 out=Path('/tmp/canonical-calibration-runtime-probe.json');out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 need(task.upload_artifact('runtime-probe',artifact_object=out,wait_on_upload=True),'probe artifact upload failed')
 print('CALIBRATION_RUNTIME_PROBE_NUMERIC_CASES_PASSED',len(result['cases']),flush=True)
if __name__=='__main__':main()
