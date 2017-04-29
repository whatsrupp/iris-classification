execfile('explore_dataset.py')
print('WELCOME TO PETAL CLASSIFIER')

def main_menu():
    print('1 - Check Accuracy of Best Model With Real Data')
    print('2 - Test All Cross Validation Models')
    print('3 - View Data Summary')
    print('4 - View Full Data Set')
    print('5 - Plot Box and Whisker Diagram')
    print('6 - Plot Histogram')
    print('7 - Plot Multivariate Scatter')
    print('8 - Exit Program')

def end_programme_message():
        print("Exiting Programme")

def make_predictions():
        execfile('make_predictions.py')

def test_models():
        execfile('test_models.py')

while True:
    main_menu()
    user_input =input()
    if user_input == 1:
        make_predictions()
    elif user_input == 2:
        test_models()
    elif user_input == 3:
        data_summary()
    elif user_input ==4:
        all_data()
    elif user_input == 5:
        plot_box_and_whisker()
    elif user_input == 6:
        plot_histogram()
    elif user_input == 7:
        plot_multivariate_scatter()
    elif user_input == 8:
        end_programme_message()
        break
    else:
        print('invalid input')
