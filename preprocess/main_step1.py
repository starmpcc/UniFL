import pandas as pd
import os, json
from transformers import AutoTokenizer
from preprocess_utils import *
import argparse


def filter_ID_TIME_NULL(src, config, rawdata_path, inputdata_path, sample=False):

    table_list = []
    column_names = {}

    for table_dict in config["Table"][src]:
        table_name = table_dict["table_name"]
        print("src : ", src, ", table_name : ", table_name)

        df = pd.read_csv(os.path.join(rawdata_path, src, table_name) + ".csv")
        if sample:
            df = df.iloc[: int(len(df) / 1000), :]

        if src == "mimic4":
            columns_upper(df)
        print("[0] Raw data shape: ", df.shape)

        # 1. Remove columns
        df.drop(columns=table_dict["time_excluded"], inplace=True)
        df.drop(columns=table_dict["id_excluded"], inplace=True)
        print("[1] Exclude useless columns: ", df.shape)

        # 2. Rename columns
        df.rename(columns={table_dict["time_column"]: "TIME"}, inplace=True)
        df.rename(columns={config["ID"][src]: "ID"}, inplace=True)
        print("[2] Rename columns to ID, TIME")

        # 3. Map ITEMID into desciprions
        if table_name in config["DICT_FILE"][src].keys():

            dict_name = config["DICT_FILE"][src][table_name][0]
            column_name = config["DICT_FILE"][src][table_name][1]
            dict_path = os.path.join(rawdata_path, src, dict_name + ".csv")
            code_dict = pd.read_csv(dict_path)

            if src == "mimic4":
                code_dict.columns = map(lambda x: str(x).upper(), code_dict.columns)
            df = name_dict(df, code_dict, column_name)
        print("[3] Map ITEMID into descriptions: ", df.shape)

        # Read ICUSTAY
        icu = pd.read_pickle(os.path.join(inputdata_path, f"{src}_cohort.pkl"))
        if src == "mimic4":
            columns_upper(icu)
        icu.rename(columns={config["ID"][src]: "ID"}, inplace=True)

        # 4. Filter ID and TIME by ICUSTAY's ID and TIME
        if src in ["mimic3", "mimic4"]:
            df = ID_time_filter_mimic(df, icu)
        else:
            df = ID_time_filter_eicu(df, icu)
        if src == "eicu":
            if table_name == "medication":
                df = eicu_med_revise(df)
            elif table_name == "infusionDrug":
                df = eicu_inf_revise(df)
        print("[4] Filter ID,TIME by ICUSTAY ID,TIME: ", df.shape)

        # 5. Filter null columns
        for col in df.columns:
            if df[col].isnull().sum() == len(df):
                df.drop(columns=col, inplace=True)
        print("[5] Filter null columns: ", df.shape)

        # 6. Filter rows where ITEMID == 'nan'
        if table_name in config["DICT_FILE"][src].keys():
            null_itemid_mask = df["ITEMID"] == "nan"
            df = df[~null_itemid_mask]
        print("[6] Filter rows where ITEMID == nan: ", df.shape, "\n")

        # Append
        df["TABLE_NAME"] = table_name
        table_list.append(df)

        column_names[table_name] = list(df.columns)

    # 7. Concat three tables
    cat_df = pd.concat(table_list, axis=0).reset_index(drop=True)

    print(
        "[7] Concat three tables: ",
        cat_df.shape,
        "=",
        table_list[0].shape,
        "+",
        table_list[1].shape,
        "+",
        table_list[2].shape,
        "=",
        (
            table_list[0].shape[0] + table_list[1].shape[0] + table_list[2].shape[0],
            table_list[0].shape[1] + table_list[1].shape[1] + table_list[2].shape[1],
        ),
    )

    # 8. Remove ID where # events < 10
    min_event = 10
    temp = pd.DataFrame(cat_df.ID.value_counts() < min_event)
    id_below10events = temp[temp["ID"]].index
    min_threshold = cat_df["ID"].isin(id_below10events)
    cat_df = cat_df[~min_threshold]
    print("[8] Remove ID where # events < 10: ", cat_df.shape)

    # 9. Sort the table
    df_sorted = cat_df.sort_values(["ID", "TIME"], ascending=True)
    print("[9] Sort the concatenated table")

    return df_sorted, column_names


def bucketize_time_gap(df_sorted):

    df_sorted["ORDER"] = list(range(len(df_sorted)))

    # 1. Bucketize time
    df_sorted["time_gap"] = df_sorted.groupby(["ID"])["TIME"].transform(
        lambda x: (x - x.shift(1)).fillna(0)
    )
    df_sorted.reset_index(drop=True, inplace=True)

    df_zero_gap = df_sorted[df_sorted["time_gap"] == 0].reset_index(drop=True)
    df_not_gap = df_sorted[df_sorted["time_gap"] != 0].reset_index(drop=True)

    df_zero_gap["time_bucket"] = "TB_0"
    df_not_gap["time_bucket"] = q_cut(df_not_gap["time_gap"], 20)
    df_not_gap["time_bucket"] = df_not_gap["time_bucket"].apply(
        lambda x: "TB_" + str(x)
    )

    df_time = pd.concat([df_zero_gap, df_not_gap], axis=0).reset_index(drop=True)
    df_time = df_time.sort_values(["ORDER"], ascending=True)

    # 2. Fill null with white space
    df_time.fillna(" ", inplace=True)
    df_time.replace("nan", " ", inplace=True)

    return df_time


def descemb_tokenize(df, table_name):

    target_cols = [
        col
        for col in df.columns
        if col not in ["ID", "TIME", "time_bucket", "time_gap", "TABLE_NAME", "ORDER"]
    ]
    table_token = tokenizer.encode(table_name)[1:-1]

    df[target_cols] = df[target_cols].applymap(
        lambda x: tokenizer.encode(round_digits(x))[1:-1] if x != " " else []
    )
    df[[col + "_dpe" for col in target_cols]] = df[target_cols].applymap(
        lambda x: make_dpe(x, number_token_list) if x != [] else []
    )

    df["event"] = df.apply(
        lambda x: sum(
            [
                tokenizer.encode(col)[1:-1] + x[col]
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )
    df["type"] = df.apply(
        lambda x: sum(
            [
                [6] * len(tokenizer.encode(col)[1:-1]) + [7] * len(x[col])
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )
    df["dpe"] = df.apply(
        lambda x: sum(
            [
                [1] * len(tokenizer.encode(col)[1:-1]) + x[col + "_dpe"]
                for col in target_cols
                if x[col] != []
            ],
            [],
        ),
        axis=1,
    )

    df["event_token"] = df.apply(
        lambda x: table_token + x["event"] + [vocab[x["time_bucket"]]], axis=1
    )
    df["type_token"] = df.apply(
        lambda x: [5] * len(table_token) + x["type"] + [4], axis=1
    )
    df["dpe_token"] = df.apply(
        lambda x: [1] * len(table_token) + x["dpe"] + [1], axis=1
    )
    return df


def col_select(df, config, src, table_name):
    selected_cols = list(config["selected"][src][table_name].keys()) + [
        "ID",
        "TIME",
        "time_bucket",
        "ORDER",
    ]
    drop_cols = [col for col in list(df.columns) if col not in selected_cols]
    df_drop = df.drop(columns=drop_cols)
    return df_drop


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rawdata_path", type=str, default="RAWDATA_PATH")
    parser.add_argument("--inputdata_path", type=str, default="INPUTDATA_PATH")
    return parser


def main():
    args = get_parser().parse_args()

    # Args
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    sample = False

    # Read config and numeric dict
    config_path = "./json/config.json"
    numeric_path = "./json/numeric_dict.json"

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    with open(numeric_path, "r") as numeric_outfile:
        numeric_dict = json.load(numeric_outfile)

    for src in ["mimic3", "eicu", "mimic4"]:

        os.makedirs(os.path.join(args.inputdata_path, src), exist_ok=True)

        last_file_pref = config["Table"][src][-1]["table_name"]
        table_names = [elem["table_name"] for elem in config["Table"][src]]

        if not os.path.isfile(
            os.path.join(args.inputdata_path, src, f"{last_file_pref}_temp.csv")
        ):

            # 1.Filter ID, TIME, NULL
            df_1st, column_names = filter_ID_TIME_NULL(
                "mimic3", config, args.rawdata_path, args.inputdata_path, sample=sample
            )
            df_temp = df_1st.copy()

            print("Buckettize time gap")
            df_time = bucketize_time_gap(df_temp)
            df = df_time.copy()

            # 2.Split cat table into three tables: LAB, PRESCRIPTIONS, INPUTEVENTS
            print("Split cat table into three tables: LAB, PRESCRIPTIONS, INPUTEVENTS")
            three_dfs = {}
            for table_name in column_names.keys():

                part_df = df[df["TABLE_NAME"] == table_name]

                table_columns = column_names[table_name] + ["time_bucket", "ORDER"]
                part_df = part_df[table_columns]
                three_dfs[table_name] = part_df

                part_df.to_csv(
                    os.path.join(args.inputdata_path, src, f"{table_name}_temp.csv")
                )
                print(table_name, list(part_df.columns))
        else:
            print("Pass 1st & 2nd starges")
            three_dfs = {}
            for table_name in table_names:
                three_dfs[table_name] = pd.read_csv(
                    os.path.join(args.inputdata_path, src, f"{table_name}_temp.csv"),
                    index_col=0,
                )
        # 3.Embed data
        quant = 20
        print("Tokenize start.")
        print(
            "It might take more than five hours in this step. Grap a cup of coffee..."
        )
        for table_dict in config["Table"][src]:
            for preprocess_type in ["whole"]:
                for embed_type in ["descemb"]:

                    table_name = table_dict["table_name"]
                    print(
                        "src : ",
                        src,
                        "table_name : ",
                        table_name,
                        "embed_type : ",
                        embed_type,
                        "preprocess_type : ",
                        preprocess_type,
                    )

                    df = three_dfs[table_name]

                    if preprocess_type == "select":
                        df = col_select(df, config, src, table_name)

                    if embed_type == "descemb":
                        df = descemb_tokenize(df, table_name)
                        df = df[
                            [
                                "ID",
                                "TIME",
                                "time_bucket",
                                "event_token",
                                "type_token",
                                "dpe_token",
                                "ORDER",
                            ]
                        ]
                        print("(EX) ", tokenizer.decode(df["event_token"].iloc[1]))

                    elif embed_type == "codeemb":
                        df = buckettize_categorize(
                            df, src, numeric_dict, table_name, quant
                        )
                        df.fillna(" ", inplace=True)
                        df.replace("nan", " ", inplace=True)
                        df = codeemb_event_merge(df, table_name)
                        df = df[
                            [
                                "ID",
                                "TIME",
                                "time_bucket",
                                "event_token",
                                "type_token",
                                "ORDER",
                            ]
                        ]

                    if not os.path.isdir(
                        os.path.join(
                            args.inputdata_path, src, f"{embed_type}_{preprocess_type}"
                        )
                    ):
                        os.makedirs(
                            os.path.join(
                                args.inputdata_path,
                                src,
                                f"{embed_type}_{preprocess_type}",
                            )
                        )

                    df.to_pickle(
                        os.path.join(
                            args.inputdata_path,
                            src,
                            f"{embed_type}_{preprocess_type}",
                            f"{table_name}.pkl",
                        )
                    )
                    print("save " + src + " " + table_name + " to pkl")


if __name__ == "__main__":
    main()
