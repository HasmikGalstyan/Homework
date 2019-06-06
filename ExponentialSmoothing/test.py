import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from holt import HoltES
from additive_seasonality import AdditiveSeasonalityES

if __name__=='__main__':
    data_energy = pd.read_csv("energy_consump.csv")
    data_energy.Date = data_energy.Date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

    data_wage = pd.read_csv("monthly-wage.csv", sep=";")
    data_wage.Month = data_wage.Month.apply(lambda x: datetime.strptime(x, "%Y-%m"))

    hes = HoltES()
    ases = AdditiveSeasonalityES(p=366)

    smoothed_energy_hes = hes.smooth(data_energy.EnergyConsump.values)
    smoothed_energy_ases = ases.smooth(data_energy.EnergyConsump.values)

    plt.plot((data_energy.Date+timedelta(hes.d)), smoothed_energy_hes, "r", label="Holt ES smoothed", alpha=0.5)
    plt.plot((data_energy.Date+timedelta(ases.d)), smoothed_energy_ases, "g", label="Additive Seasonality ES smoothed",
             alpha=0.5)
    plt.plot(data_energy.Date, data_energy.EnergyConsump.values, label="real", alpha=0.5)
    plt.legend()
    plt.title('energy consumption versus time')
    plt.show()

    hes = HoltES()
    ases = AdditiveSeasonalityES(p=12)

    smoothed_wage_hes = hes.smooth(data_wage['Real wage'])
    smoothed_wage_ases = ases.smooth(data_wage['Real wage'])

    plt.plot((data_wage.Month.apply(lambda x: x+relativedelta(years=hes.d//12, months=hes.d%12))),
             smoothed_wage_hes, "r", label="Holt ES smoothed", alpha=0.5)
    plt.plot((data_wage.Month.apply(lambda x: x+relativedelta(years=ases.d//12, months=ases.d%12))),
             smoothed_wage_ases, "g", label="Additive Seasonality ES smoothed",alpha=0.5)
    plt.plot(data_wage.Month, data_wage['Real wage'], label="real", alpha=0.5)
    plt.legend()
    plt.title('wage versus time')
    plt.show()
