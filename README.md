# Linear_endless_bandits

Repository of the paper "Orthogonal Function Representations for Continuous
Armed Bandits"

To perform the experiment reported in the plots (figures 2 and 3) run the command
python main.py

It will be requested which expetiment to run (a,b,c,d) and if you want to have even basis (figure 3) or no (figure 2).
At the end of the experiment a plot of the estimated cumulative regret is shown, and training data are saved in folder "results",
together with a .pdf plot of the reagret and a .json containing the running times.

N.B., to modify the time horizon or the number of seeds change variables T, seeds of the file main.py. Default values are
T = 1000
seeds = 5
to make the computation fast.
