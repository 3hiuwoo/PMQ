from utils import start_logging, stop_logging

if __name__ == '__main__':
    print('=> test')
    start_logging(0, 'logs')
    print('=> something')
    stop_logging()
    print('=> done')