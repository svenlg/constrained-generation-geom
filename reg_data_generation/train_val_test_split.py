import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Example: load your DataFrame (replace with your actual data)

def main(args):
    
    df_path = f"data/{args.experiment}/results.csv"
    df = pd.read_csv(df_path)

    assert args.train + args.val <= 1, "Train and validation proportions must sum to less than or equal to 1"
    seed = args.seed

    # First split: train vs temp
    train_df, temp_df = train_test_split(df, train_size=args.train, random_state=seed, shuffle=True)

    # Second split: val (10%) vs test (10%) from the temp set
    val_fraction = args.val / (1 - args.train)
    val_df, test_df = train_test_split(temp_df, train_size=val_fraction, random_state=seed, shuffle=True)

    # Save them to CSV (without index)
    train_df.to_csv(f"data/{args.experiment}/train.csv", index=False)
    val_df.to_csv(f"data/{args.experiment}/val.csv", index=False)
    test_df.to_csv(f"data/{args.experiment}/test.csv", index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets")
    parser.add_argument("-e", "--experiment", type=str, required=True,
                        help="experiment to use")
    parser.add_argument("--train", type=float, default=0.8, 
                        help="Proportion of the dataset to include in the train split")
    parser.add_argument("--val", type=float, default=0.1,
                        help="Proportion of the dataset to include in the val split")
    parser.add_argument("--seed", type=int, default=0, 
                        help="Random seed for reproducibility")
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())