General configuration for all yaml files:
- data_path: '' (The path to the data, different per dataset)
- input_size: 6
- window: 60
- overlap: 60 (Apparently, this means no overlapping is employed)
- batch_size: 64
- use_index_as_label: True (For DIET, it returns the data index as label)
- use_val_with_train: True (It forces the validation set to be part of the train set)

Other considerations:
- All datamodules used come from minerva.data.data_modules.har_rodrigues_24.HARDataModuleCPC
- The datasets employed: kuhar, motionsense, realworld thigh, realworld waist, wisdm, uci, recodgait, and hapt. 