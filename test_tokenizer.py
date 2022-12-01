def TEST_TOKENIZER_ON_ORIGINAL_DATA(dataset, tokenized_data):
    for i in range(len(dataset)):
        og_input_data, og_output_data = dataset[i]
        decoded_data_input = tokenizer.decode_inputs(tokenized_data[i][0])
        decoded_data_input = " ".join(decoded_data_input)

        decoded_data_output = tokenizer.decode_outputs(tokenized_data[i][1])
        decoded_data_output = " ".join(decoded_data_output)
        og_input_data = og_input_data + " EOS SOS"
        og_output_data = og_output_data + " EOS"
        if (og_input_data != decoded_data_input):
            print("FAIL: TRAIN DATA")
            print("Expected: ", og_input_data)
            print("Output", decoded_data_input)
            return False

        if (og_output_data != decoded_data_output):
            print("FAIL: OUTPUT DATA")
            print("Expected: ", og_output_data)
            print("Output", decoded_data_output)
            return False
    return True


