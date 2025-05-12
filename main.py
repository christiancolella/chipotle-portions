import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

def simulate_n_days(n_days: int, n_daily_workers: int, portions: np.ndarray):
    # Ensure correct input
    n_total_workers = portions.shape[0]
    assert n_total_workers >= n_daily_workers
    
    # Compute schedule matrix
    schedule = np.zeros((n_days, n_total_workers))
    
    for i in range(n_days):
        daily_workers = np.random.choice(n_total_workers, size=n_daily_workers, replace=False)
        schedule[i, daily_workers] = 1
    
    # Compute consumption (target) vector
    n_customers = np.random.normal(1, 0.1, (n_days))
    consumption = n_customers / n_daily_workers * np.matmul(schedule, portions)
    
    return consumption, schedule

def get_coefficients(y_total, x_total):
    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    # Create and fit the model
    model = Lasso(alpha=0.1, max_iter=5000)
    model.fit(x_train, y_train)
    
    # Return coefficients
    return np.array(model.coef_)
    
def run_one_simulation(n_days, n_daily_workers, n_total_workers):
    portions = np.random.normal(113, 15, (n_total_workers))
    true_argmax = np.argmax(portions)
    
    # print('True portions:', portions)
    # print('True argmax:', true_argmax)
    
    y_total, x_total = simulate_n_days(n_days, n_daily_workers, portions)
    
    # Create dataframe
    # arr = np.hstack((y_total.reshape(-1, 1), x_total))
    # columns = ["consumption"] + [f"worker_{i + 1}" for i in range(n_total_workers)]
    
    # df = pd.DataFrame(arr, columns=columns)
    
    coefs = get_coefficients(y_total, x_total)
    predicted_argmax = np.argmax(coefs)
    
    # print('Predicted portions:', coefs)
    # print('Predicted argmax:', predicted_argmax)
    
    # if true_argmax == predicted_argmax:
    #     print('**Correct**\n')
    # else:
    #     print('**Incorrect**\n')
    
    return true_argmax == predicted_argmax

def run_n_simulations(n_sims, n_days, n_daily_workers, n_total_workers):
    n_correct = 0
    
    for i in range(n_sims):
        # print(f'[Simulation {i + 1}/{n_sims}]')
        result = run_one_simulation(n_days, n_daily_workers, n_total_workers)
        
        if result:
            n_correct += 1
    
    accuracy = n_correct / n_sims
    
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

def create_scatterplot(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', alpha=0.7, edgecolors='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def anlayze_accuracy():
    N_SIMS = 100
    N_DAILY_WORKERS = 3
    N_TOTAL_WORKERS = 20
    
    # Allow n_days to vary
    min_n_days = 100
    max_n_days = 600
    
    # Scatterplot data
    x = np.array(range(min_n_days, max_n_days))
    y = np.zeros((len(x)))
    
    for i, n_days in enumerate(x):
        print(f'N_DAYS={n_days}')
        y[i] = run_n_simulations(N_SIMS, n_days, N_DAILY_WORKERS, N_TOTAL_WORKERS)
        
    create_scatterplot(x, y, 'n_days', 'Accuracy', 'Change in accuracy vs. number of days with data')
    
def simulate_chipotle():
    N_DAYS = 20                 # 20 shifts are evaluated
    N_DAILY_WORKERS = 3         # 3 food-assembly workers per shift
    N_TOTAL_WORKERS = 20        # 20 total workers
    
    N_EPOCHS = 100              # Number of times the worst-performing worker is replaced
    
    # Initial portions
    portions = np.random.normal(113, 25, (N_TOTAL_WORKERS))
    
    # Data for scatterplot
    x = np.array(range(N_EPOCHS))
    y = np.zeros((len(x)))
    
    # Early stopping
    best_loss = float('infinity')
    patience = 10
    counter = 0
    
    for i in range(N_EPOCHS):
        print(f'[Epoch {i + 1}]')
        
        # Run simulation
        y_total, x_total = simulate_n_days(N_DAYS, N_DAILY_WORKERS, portions)
        
        # Compute loss
        loss = np.average(y_total) - 113
        y[i] = loss
        print(f'Loss: {loss:.4f}\n')
    
        # Replace worst-performing worker
        coefs = get_coefficients(y_total, x_total)
        argmax = np.argmax(coefs)
        
        if loss < best_loss:
            best_loss = loss
            counter = 0
            
        else:
            counter += 1
            
            if counter >= patience:
                print(f'Early stopping after {i + 1} epochs...\n')
                break
        
        print(f'Replacing worker {argmax + 1}...')
        print(f'Old portion size: {portions[argmax]}')
        
        portions[argmax] = np.random.normal(113, 25)
        print(f'New portion size: {portions[argmax]}\n')
        
    create_scatterplot(x, y, 'Epoch', 'Loss', 'Change in loss vs. number of replacement iterations')
    
def main():
    simulate_chipotle()

if __name__ == '__main__':
    main()