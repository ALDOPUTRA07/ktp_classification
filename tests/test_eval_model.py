from ktp_classification.eval_model import print_train_time


def test_print_train_time():
    # act
    time = print_train_time(1, 10)

    # assert
    assert time == 9
