#!/usr/bin/python3

"""Program checking the grade repartition in every classes at Poudlard in the dataset train file"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


input_file = "dataset/dataset_train.csv"

if __name__ == '__main__':
    try:
        dataset = pd.read_csv(input_file, delimiter=",")
    except Exception as e:
        print("Can't open the file passed as argument, program will exit")
        exit(e)

    Ravenclaw = dataset[(dataset['Hogwarts House'] == 'Ravenclaw')]
    Slytherin = dataset[(dataset['Hogwarts House'] == 'Slytherin')]
    Gryffindor = dataset[(dataset['Hogwarts House'] == 'Gryffindor')]
    Hufflepuff = dataset[(dataset['Hogwarts House'] == 'Hufflepuff')]

    Ravenclaw_mean_Arithmancy = Ravenclaw['Arithmancy'].mean()
    Ravenclaw_mean_Astronomy = Ravenclaw['Astronomy'].mean()
    Ravenclaw_mean_Herbology = Ravenclaw['Herbology'].mean()
    Ravenclaw_mean_DADA = Ravenclaw['Defense Against the Dark Arts'].mean()
    Ravenclaw_mean_Divination = Ravenclaw['Divination'].mean()
    Ravenclaw_mean_MS = Ravenclaw['Muggle Studies'].mean()
    Ravenclaw_mean_AR = Ravenclaw['Ancient Runes'].mean()
    Ravenclaw_mean_HM = Ravenclaw['History of Magic'].mean()
    Ravenclaw_mean_Transfiguration = Ravenclaw['Transfiguration'].mean()
    Ravenclaw_mean_Potions = Ravenclaw['Potions'].mean()
    Ravenclaw_mean_CMC = Ravenclaw['Care of Magical Creatures'].mean()
    Ravenclaw_mean_Charms = Ravenclaw['Charms'].mean()
    Ravenclaw_mean_Flying = Ravenclaw['Flying'].mean()

    Slytherin_mean_Arithmancy = Slytherin['Arithmancy'].mean()
    Slytherin_mean_Astronomy = Slytherin['Astronomy'].mean()
    Slytherin_mean_Herbology = Slytherin['Herbology'].mean()
    Slytherin_mean_DADA = Slytherin['Defense Against the Dark Arts'].mean()
    Slytherin_mean_Divination = Slytherin['Divination'].mean()
    Slytherin_mean_MS = Slytherin['Muggle Studies'].mean()
    Slytherin_mean_AR = Slytherin['Ancient Runes'].mean()
    Slytherin_mean_HM = Slytherin['History of Magic'].mean()
    Slytherin_mean_Transfiguration = Slytherin['Transfiguration'].mean()
    Slytherin_mean_Potions = Slytherin['Potions'].mean()
    Slytherin_mean_CMC = Slytherin['Care of Magical Creatures'].mean()
    Slytherin_mean_Charms = Slytherin['Charms'].mean()
    Slytherin_mean_Flying = Slytherin['Flying'].mean()

    Gryffindor_mean_Arithmancy = Gryffindor['Arithmancy'].mean()
    Gryffindor_mean_Astronomy = Gryffindor['Astronomy'].mean()
    Gryffindor_mean_Herbology = Gryffindor['Herbology'].mean()
    Gryffindor_mean_DADA = Gryffindor['Defense Against the Dark Arts'].mean()
    Gryffindor_mean_Divination = Gryffindor['Divination'].mean()
    Gryffindor_mean_MS = Gryffindor['Muggle Studies'].mean()
    Gryffindor_mean_AR = Gryffindor['Ancient Runes'].mean()
    Gryffindor_mean_HM = Gryffindor['History of Magic'].mean()
    Gryffindor_mean_Transfiguration = Gryffindor['Transfiguration'].mean()
    Gryffindor_mean_Potions = Gryffindor['Potions'].mean()
    Gryffindor_mean_CMC = Gryffindor['Care of Magical Creatures'].mean()
    Gryffindor_mean_Charms = Gryffindor['Charms'].mean()
    Gryffindor_mean_Flying = Gryffindor['Flying'].mean()

    Hufflepuff_mean_Arithmancy = Hufflepuff['Arithmancy'].mean()
    Hufflepuff_mean_Astronomy = Hufflepuff['Astronomy'].mean()
    Hufflepuff_mean_Herbology = Hufflepuff['Herbology'].mean()
    Hufflepuff_mean_DADA = Hufflepuff['Defense Against the Dark Arts'].mean()
    Hufflepuff_mean_Divination = Hufflepuff['Divination'].mean()
    Hufflepuff_mean_MS = Hufflepuff['Muggle Studies'].mean()
    Hufflepuff_mean_AR = Hufflepuff['Ancient Runes'].mean()
    Hufflepuff_mean_HM = Hufflepuff['History of Magic'].mean()
    Hufflepuff_mean_Transfiguration = Hufflepuff['Transfiguration'].mean()
    Hufflepuff_mean_Potions = Hufflepuff['Potions'].mean()
    Hufflepuff_mean_CMC = Hufflepuff['Care of Magical Creatures'].mean()
    Hufflepuff_mean_Charms = Hufflepuff['Charms'].mean()
    Hufflepuff_mean_Flying = Hufflepuff['Flying'].mean()

    Arithmancy_score = np.nanstd([Hufflepuff_mean_Arithmancy, Gryffindor_mean_Arithmancy,
                                  Ravenclaw_mean_Arithmancy, Slytherin_mean_Arithmancy])
    Astronomy_score = np.nanstd([Hufflepuff_mean_Astronomy, Gryffindor_mean_Astronomy,
                                 Ravenclaw_mean_Astronomy, Slytherin_mean_Astronomy])
    Herbology_score = np.nanstd([Hufflepuff_mean_Herbology, Gryffindor_mean_Herbology,
                                 Ravenclaw_mean_Herbology, Slytherin_mean_Herbology])
    DADA_score = np.nanstd([Hufflepuff_mean_DADA, Gryffindor_mean_DADA,
                            Ravenclaw_mean_DADA, Slytherin_mean_DADA])
    Divination_score = np.nanstd([Hufflepuff_mean_Divination, Gryffindor_mean_Divination,
                                  Ravenclaw_mean_Divination, Slytherin_mean_Divination])
    MS_score = np.nanstd([Hufflepuff_mean_MS, Gryffindor_mean_MS,
                          Ravenclaw_mean_MS, Slytherin_mean_MS])
    AR_score = np.nanstd([Hufflepuff_mean_AR, Gryffindor_mean_AR,
                          Ravenclaw_mean_AR, Slytherin_mean_AR])
    HM_score = np.nanstd([Hufflepuff_mean_HM, Gryffindor_mean_HM,
                          Ravenclaw_mean_HM, Slytherin_mean_HM])
    Transfiguration_score = np.nanstd([Hufflepuff_mean_Transfiguration, Gryffindor_mean_Transfiguration,
                                       Ravenclaw_mean_Transfiguration, Slytherin_mean_Transfiguration])
    Potions_score = np.nanstd([Hufflepuff_mean_Potions, Gryffindor_mean_Potions,
                               Ravenclaw_mean_Potions, Slytherin_mean_Potions])
    CMC_score = np.nanstd([Hufflepuff_mean_CMC, Gryffindor_mean_CMC,
                           Ravenclaw_mean_CMC, Slytherin_mean_CMC])
    Charms_score = np.nanstd([Hufflepuff_mean_Charms, Gryffindor_mean_Charms,
                              Ravenclaw_mean_Charms, Slytherin_mean_Charms])
    Flying_score = np.nanstd([Hufflepuff_mean_Flying, Gryffindor_mean_Flying,
                              Ravenclaw_mean_Flying, Slytherin_mean_Flying])

    array = [Arithmancy_score, Astronomy_score, Herbology_score, DADA_score, Divination_score, MS_score,
             AR_score, HM_score, Transfiguration_score, Potions_score, CMC_score, Charms_score,
             Flying_score]
    x = np.arange(13)
    plt.bar(x, array)
    plt.xticks(x, ['ARI', 'AST', 'HER', 'DADA', 'DIV', 'MS',
             'AR', 'HM', 'TRA', 'POT', 'CMC', 'CHA',
             'FLY'])
    plt.show()