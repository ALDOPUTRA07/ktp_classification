import multiprocessing
import os
import time

"""
    function for start Flask server
"""


def run_fastapi_server():
    os.system('bash serve/run.sh')


"""
    function for start Streamlit server
"""


def run_streamlit_server():
    time.sleep(4)
    os.system('bash streamlit_app/start.sh')


if __name__ == '__main__':
    """
    Run Backend Server & Frontend Server at the same time.
    """
    p1 = multiprocessing.Process(target=run_fastapi_server)
    p2 = multiprocessing.Process(target=run_streamlit_server)

    p1.start()
    p2.start()

    p1.join()
    p2.join()
