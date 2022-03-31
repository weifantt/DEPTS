f = open('command.sh','w')

for data, split, J in [('electricity','deepar',4),('electricity','last',32)]:
    for seed in [1,2,3,4,5]:
        for lookback in [2,3,4,5,6,7]:
            f.write(f"python main_depts.py --dataset {data} --split {split} --seed {seed} --lookback {lookback} --J {J} --output output/{data}/seed:{seed},lookback:{lookback}\n")
    f.write("\n\n\n")
#f.write("\n\n\n")


for data, split, J in [('traffic','deepar',8),('traffic','last',8)]:
    for seed in [1,2,3,4,5]:
        for lookback in [2,3,4,5,6,7]:
            f.write(f"python main_depts.py --dataset {data} --split {split} --seed {seed} --lookback {lookback} --J {J} --output output/{data}/seed:{seed},lookback:{lookback}\n")
    f.write("\n\n\n")
#f.write("\n\n\n")


for data, split, J in [('caiso','last18months',8),('caiso','last15months',32),('caiso','last12months',32),('caiso','last9months',8)]:
    for seed in [1,2,3,4,5]:
        for lookback in [2,3,4,5,6,7]:
            f.write(f"python main_depts.py --dataset {data} --split {split} --seed {seed} --lookback {lookback} --J {J} --output output/{data}/seed:{seed},lookback:{lookback}\n")
    f.write("\n\n\n")


for data, split, J in [('product','last12months',8),('product','last9months',8),('product','last6months',32),('product','last3months',32)]:
    for seed in [1,2,3,4,5]:
        for lookback in [2,3,4,5,6,7]:
            f.write(f"python main_depts.py --dataset {data} --split {split} --seed {seed} --lookback {lookback} --J {J} --output output/{data}/seed:{seed},lookback:{lookback}\n")
    f.write("\n\n\n")