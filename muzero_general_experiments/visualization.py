from tensorboard.backend.event_processing import event_accumulator
import pandas as pd

# tf.logging.set_verbosity(tf.logging.ERROR)

basedir = ""


def load_tf(dirname):

    ea = event_accumulator.EventAccumulator(dirname, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    dframes = {}
    mnames = ea.Tags()['scalars']

    for n in mnames:
        dframes[n] = pd.DataFrame(ea.Scalars(n), columns=["wall_time", "epoch", n.replace('val/', '')])
        dframes[n].drop("wall_time", axis=1, inplace=True)
        dframes[n] = dframes[n].set_index("epoch")
    return pd.concat([v for k, v in dframes.items()], axis=1)

df = load_tf('../result_muzero/default_cartpole')
# def load_tf_jobs(regex):
#     prefix = basedir + "results/"
#     job_dirs = glob.glob(prefix + regex)
#
#     rows = []
#     for job in job_dirs:
#         job_name = os.path.basename(os.path.normpath(job))
#
#         # this loads in all the hyperparams from another file,
#         # do your own thing here instead
#         options = load_json(job + '/opt.json')
#         try:
#             results = load_tf(job.replace(prefix, ''))
#         except:
#             continue
#
#         for opt in options:
#             results[opt] = options[opt]
#         rows.append(results)
#
#     for row in rows:
#         row['epoch'] = row.index
#         row.reset_index(drop=True, inplace=True)
#     df = pd.concat(rows)
#     return df

# experiment_id = 'results/cartpole/2020-10-05--18-30-37'
#
# import tensorboard as tb
#
# data = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = data.get_scalars()
print(df)