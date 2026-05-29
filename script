import numpy as np
import os

# your folder
path = "exp/ngc_transformer/component/custom"

# go through all .npz files
for file in sorted(os.listdir(path)):

    if file.endswith(".npz"):

        print("\n" + "="*100)
        print("FILE:", file)
        print("="*100)

        # load file
        data = np.load(os.path.join(path, file))

        # print everything inside
        for key in data.files:

            print("\nKEY:", key)
            print("-"*80)

            # print actual content
            print(data[key])

            print("-"*80)
