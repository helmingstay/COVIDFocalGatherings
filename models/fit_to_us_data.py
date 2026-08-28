# %%
import numpy as np
import pandas as pd
from time import time
import torch
from models.extract_csv import read_datafiles
from models.transmission_model import GeneralSpatialTransmissionModel

torch.set_num_threads(10)


#%% Read and prepare COVID and demographic data
import models.read_config_file as rc

#%% Specify model parameters
# Note this is not the best way to process data
epi_data = rc.data_dict["covid_data"]
epi_data["isoweek"] = pd.to_datetime(epi_data["isoweek"])
epi_data = (epi_data[epi_data["fips"].isin(rc.fips_codes)]
            .set_index(["fips", "isoweek"])
            ["deaths"]
            .unstack(level=0)
            .fillna(0))
epi_data.index = (epi_data.index - rc.origin_date).days.values

epi_data = epi_data.loc[:rc.fixed_par_dict["tf"]]

#%% Create model
rng = np.random.default_rng(seed=2398424384)
n_particles = 1000
n_mif = 50

model = GeneralSpatialTransmissionModel(**rc.fixed_par_dict,
                                        **rc.model_tensors_dict,
                                        n_particles=n_particles,
                                        numpy_rng=rng)


#%% Simulate model

pars = list(rc.par_dict.values())  # Dicts preserve order

#%%
# Iterated filtering test
start = time()
y0 = model.initialise_model()
par0 = pars.copy()
#par0[0] *= 1.03

y_out3, obs_out3, par_out3 = model.simulate_euler_forward(y0=y0,
                                                          t_eval=epi_data.index.values,
                                                          y_obs=torch.tensor(epi_data.values),
                                                          pars=par0,
                                                          Nmif=n_mif)
end = time()
print(end - start)


#%% Simulate at MLE

par_mle = par_out3[-1, :]
start = time()
y0 = model.initialise_model()
y_out, obs_out = model.simulate_euler_forward(y0=y0,
                                              t_eval=epi_data.index.values,
                                              pars=par_mle)
end = time()
print(end - start)

#%%
import seaborn as sns

mle_dict = pd.Series(dict(zip(rc.par_dict.keys(), par_mle.numpy())))
df_par_out = pd.DataFrame(par_out3, columns=rc.par_dict.keys())
abc = list("ABCDEFG")


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=3)
for i, p in enumerate(["R0", 'local_coef', 'fixed_reduction_coef']):
    ax = axes[i]
    ax.plot(df_par_out[p])
    ax.set_ylabel(p)
    ax.axhline(mle_dict[p], linestyle="--")
    ax.set_title(abc[i], loc="left")

axes[-1].set_xlabel("Data point")
fig.tight_layout()
sns.despine(trim=False)
fig.savefig("./figures/fig_traceplot.pdf")
plt.show()


#%% Create a dataframe from simulation output for each model compartment
# then stack

def get_output_dataframe(y_out, obs_out):
    def get_data_frames(output_data, compartment_name):
        return (pd.concat([pd.DataFrame(output_data[:, r],
                                        index=pd.Series(epi_data.index.values, name="day"),
                                        columns=model.location_names["name"]).assign(run=r)
                           for r in range(n_particles)])
                .assign(compartment=compartment_name))

    df_y_out = pd.concat([get_data_frames(y_out[:, :, i], c)
                          for i, c in enumerate(model.compartment_names)])

    df_y_out = pd.concat([df_y_out, get_data_frames(obs_out, "obs")])

    # Calculate data time
    df_y_out["date"] = rc.origin_date + pd.to_timedelta(df_y_out.index, "days")

    return df_y_out

df_y_out = get_output_dataframe(y_out, obs_out)


df_new_cases = df_y_out[df_y_out["compartment"] == "obs"].iloc[:, :-2]



df_new_cases.to_csv("./output_data/simulated_new_cases.csv")
