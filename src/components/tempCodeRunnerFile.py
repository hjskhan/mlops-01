                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info('reading of train test data complete')
                logging.info('obtaining preprocessing object')

                preprocessing_obj = self.get_data_transformer_object()
                target_column_name = 'math_score'
                numerical_columns = ['writing_score', 'reading_score']

                input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df = test_df[target_column_name]
