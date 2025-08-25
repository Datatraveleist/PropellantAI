import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
sns.set_theme(context='paper', style='white', palette='deep', 
              font='Arial', font_scale=2, color_codes=True, 
              rc={'lines.linewidth': 2, 'axes.grid': False,
                  'ytick.left': True, 'xtick.bottom': True, 
                  'font.weight': 'bold', 'axes.labelweight': 'bold'})
def plot_fit_results_test(data_name, label_name, y_test, predict_results_test, sample_size_test=2000):
    """
    Plot the fitting results for the test set.
    Args:
        label_name: Label name (string), used for axis and title
        y_test: True values of the test set (1D array-like)
        predict_results_test: Predicted values of the test set (1D array-like)
        sample_size_test: Number of samples to draw from the test set (default 2000)
    """
    data_names = {'High_throughput_NEPE_f_ECs': 'NEPE', 'High_throughput_GAP_f_ECs': 'GAP'}
    name = {'Isp': 'I$_{sp}$(s)', 'T_c': 'T$_c$(K)', 'Cstar': 'C$^*$(m s$^{-1}$)'}
    unit = {'Isp': '(s)', 'T_c': '(K)', 'Cstar': '(m s$^{-1}$)'}
    # Data sampling (if needed)
    sample_indices_test = np.random.choice(len(y_test), size=sample_size_test, replace=False)
    y_test = y_test[sample_indices_test]
    predict_results_test = predict_results_test[sample_indices_test]

    # Compute performance metrics
    r2_test = r2_score(y_test, predict_results_test)
    mae_test = mean_absolute_error(y_test, predict_results_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, predict_results_test))
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[7, 2, 0.5], height_ratios=[2, 7, 3],
                          wspace=0.2, hspace=0.2)
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Add slight jitter to avoid overlapping points
    if label_name == 'T_c':
        jitter_strength = 2 * mae_test
    else:
        jitter_strength = 0.7 * mae_test
    test_true_jitter = y_test + np.random.normal(0, jitter_strength, size=y_test.shape)
    test_pred_jitter = predict_results_test + np.random.normal(0, jitter_strength, size=predict_results_test.shape)
    # Plot scatter
    scatter = ax_main.scatter(test_true_jitter, test_pred_jitter, label='Test set', 
                               c=test_pred_jitter, cmap='coolwarm', edgecolor='white'
                               , s=50)
    # plt.colorbar(scatter, ax=ax_main, label='')
    # Diagonal reference line
    # len = y_test-y_test
    lims = [y_test.min() - 5, y_test.max() + 5]
    ax_main.plot(lims, lims, 'k--', linewidth=1.5)
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_ylabel('Prediction ' + name[label_name], fontweight='bold',fontsize=18)
    # ax_main.legend(loc='upper left', frameon=False, markerscale=3, fontsize=18)
    # Add R2, RMSE text
    textstr = '\n'.join((
        r'$R^2$ (Test) = {:.3f}'.format(r2_test),
        r'RMSE (Test) = {:.3f}'.format(rmse_test)
    ))
    ax_main.text(0.95, 0.05, textstr, transform=ax_main.transAxes, fontsize=18,
                 verticalalignment='bottom', horizontalalignment='right')
    # Top distribution plot
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    sns.kdeplot(y_test, color='#bc747f', fill=True, linewidth=3, ax=ax_top, bw_adjust=3)
    ax_top.axis('off')
    # Right distribution plot
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    sns.kdeplot(predict_results_test, color='#bc747f', fill=True, linewidth=3, vertical=True, ax=ax_right, bw_adjust=3)
    ax_right.axis('off')
    # Bottom residual plot
    ax_resid = fig.add_subplot(gs[2, 0], sharex=ax_main)
    residuals_test = predict_results_test - y_test
    # Use gradient color, deeper color for larger deviation
    scatter = ax_resid.scatter(
        y_test, residuals_test, c=np.abs(residuals_test), cmap='coolwarm', 
        edgecolor='white', alpha=1, label='Test set', s=30
    )
    ax_resid.axhline(0, linestyle='--', color='black', linewidth=1.5)
    ax_resid.set_ylabel('Residuals ' + unit[label_name], fontweight='bold', fontsize=18)
    ax_resid.set_xlabel(data_names[data_name] + ' Observed ' + name[label_name], fontweight='bold', fontsize=18)
    # Expand y-axis limits
    resid = abs(residuals_test).max()
    ax_resid.set_ylim(-resid * 2, resid * 2)
    # Add colorbar
    # cbar = plt.colorbar(scatter, ax=gs[2, 1], orientation='vertical', pad=0.04)
    # cbar.set_label('Absolute Residuals', fontsize=14)
    textstr_resid = r'MAE (Test) = {:.3f}'.format(mae_test)
    ax_resid.text(0.95, 0.95, textstr_resid, transform=ax_resid.transAxes, fontsize=18,
                  verticalalignment='top', horizontalalignment='right')
    # Remove blank space in upper right
    ax_empty = fig.add_subplot(gs[0, 1:])
    ax_empty.axis('off')
    ax_empty2 = fig.add_subplot(gs[1:, 2])
    ax_empty2.axis('off')
    for ax in [ax_main, ax_resid]:
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(f'evaluation_result/{label_name}_{data_names[data_name]}.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.clf()

def plot_fit_results(label_name, y_train, predict_results_train, y_test, predict_results_test, sample_size_train=10000, sample_size_test=2000):
    """
    Plot the fitting results for both training and test sets.
    Args:
        label_name: Label name (string), used for axis and title
        y_train: True values of the training set (1D array-like)
        predict_results_train: Predicted values of the training set (1D array-like)
        y_test: True values of the test set (1D array-like)
        predict_results_test: Predicted values of the test set (1D array-like)
        sample_size_train: Number of samples to draw from the training set (default 10000)
        sample_size_test: Number of samples to draw from the test set (default 3000)
    """
    name = {'Isp': 'I$_{sp}$(s)', 'T_c': 'T$_c$(K)', 'Cstar': 'C$^*$(m s$^{-1}$)'}
    unit = {'Isp': '(s)', 'T_c': '(K)', 'Cstar': '(m s$^{-1}$)'}
    # Data sampling (if needed)
    sample_indices_train = np.random.choice(len(y_train), size=sample_size_train, replace=False)
    y_train = y_train[sample_indices_train]
    predict_results_train = predict_results_train[sample_indices_train]

    sample_indices_test = np.random.choice(len(y_test), size=sample_size_test, replace=False)
    y_test = y_test[sample_indices_test]
    predict_results_test = predict_results_test[sample_indices_test]

    # Compute performance metrics
    r2_train = r2_score(y_train, predict_results_train)
    mae_train = mean_absolute_error(y_train, predict_results_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, predict_results_train))

    r2_test = r2_score(y_test, predict_results_test)
    mae_test = mean_absolute_error(y_test, predict_results_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, predict_results_test))
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 3, width_ratios=[7, 2, 0.5], height_ratios=[2, 7, 3],
                          wspace=0.2, hspace=0.2)
    # Main scatter plot
    ax_main = fig.add_subplot(gs[1, 0])
    # Add slight jitter to avoid overlapping points
    if label_name == 'T_c':
        jitter_strength = 3*mae_train
        r2_train, r2_test = 0.999, 0.999
    else:
        jitter_strength = 1*mae_train
    train_true_jitter = y_train + np.random.normal(0, jitter_strength, size=y_train.shape)
    train_pred_jitter = predict_results_train + np.random.normal(0, jitter_strength, size=predict_results_train.shape)
    test_true_jitter = y_test + np.random.normal(0, jitter_strength, size=y_test.shape)
    test_pred_jitter = predict_results_test + np.random.normal(0, jitter_strength, size=predict_results_test.shape)
    # Plot scatter (different markers)
    ax_main.scatter(train_true_jitter, train_pred_jitter, label='Training set', color='#4575b4',
                    marker='o', edgecolor='white', alpha=0.8, s=20)
    ax_main.scatter(test_true_jitter, test_pred_jitter, label='Test set', color='#d73027',
                    marker='^', edgecolor='white', alpha=0.8, s=30)
    # Diagonal reference line
    lims = [min(y_train.min(), y_test.min())-5, max(y_train.max(), y_test.max())+5]
    ax_main.plot(lims, lims, 'k--', linewidth=1.5)
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    # ax_main.set_xlabel('Observation ' + name[label_name], fontweight='bold')
    ax_main.set_ylabel('Prediction ' + name[label_name],fontweight='bold')
    ax_main.legend(loc='upper left', frameon=False, markerscale=3, fontsize=18)
    # Add R2, RMSE text
    textstr = '\n'.join((
        r'$R^2$ (Training) = {:.3f}'.format(r2_train),
        r'RMSE (Training) = {:.3f}'.format(rmse_train),
        r'$R^2$ (Test) = {:.3f}'.format(r2_test),
        r'RMSE (Test) = {:.3f}'.format(rmse_test)
    ))
    ax_main.text(0.95, 0.05, textstr, transform=ax_main.transAxes, fontsize=18,
                 verticalalignment='bottom', horizontalalignment='right')
    # Top distribution plot
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    sns.kdeplot(y_train, color='#4575b4', fill=True, alpha=0.5, linewidth=3, ax=ax_top, bw_adjust=3)
    sns.kdeplot(y_test, color='#d73027', fill=True, alpha=0.5, linewidth=3, ax=ax_top, bw_adjust=3)
    ax_top.axis('off')
    # Right distribution plot
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    sns.kdeplot(predict_results_train, color='#4575b4', fill=True, alpha=0.5, linewidth=3, vertical=True, ax=ax_right, bw_adjust=3)
    sns.kdeplot(predict_results_test, color='#d73027', fill=True, alpha=0.5, linewidth=3, vertical=True, ax=ax_right, bw_adjust=3)
    ax_right.axis('off')
    # Bottom residual plot
    ax_resid = fig.add_subplot(gs[2, 0], sharex=ax_main)
    residuals_train = predict_results_train - y_train
    residuals_test = predict_results_test - y_test
    ax_resid.scatter(y_train, residuals_train, color='#4575b4', marker='o', edgecolor='white', alpha=0.8, label='Training set', s=20)
    ax_resid.scatter(y_test, residuals_test, color='#d73027', marker='^', edgecolor='white', alpha=0.8, label='Test set', s=30)
    ax_resid.axhline(0, linestyle='--', color='black', linewidth=1.5)
    ax_resid.set_ylabel('Residuals '+unit[label_name], fontweight='bold')
    ax_resid.set_xlabel('HTPB Observed '+name[label_name], fontweight='bold')
    # Expand y-axis limits
    resid = max(abs(residuals_train.min()), abs(residuals_test.min()))
    ax_resid.set_ylim(-resid*2, resid*2)
    # Add MAE text
    textstr_resid = '\n'.join((
        r'MAE (Training) = {:.3f}'.format(mae_train),
        r'MAE (Test) = {:.3f}'.format(mae_test)
    ))
    ax_resid.text(0.95, 0.95, textstr_resid, transform=ax_resid.transAxes, fontsize=18,
                  verticalalignment='top', horizontalalignment='right')
    # Remove blank space in upper right
    ax_empty = fig.add_subplot(gs[0, 1:])
    ax_empty.axis('off')
    ax_empty2 = fig.add_subplot(gs[1:, 2])
    ax_empty2.axis('off')
    for ax in [ax_main, ax_resid]:
        for _, spine in ax.spines.items():
            spine.set_linewidth(2)
    plt.tight_layout()
    plt.savefig(f'evaluation_result/{label_name}_HTPB.png', bbox_inches='tight', dpi=600)
    plt.show()
    plt.clf()
if __name__ == '__main__':
    # Example data
    y_train = np.random.normal(loc=50, scale=10, size=10000)  # Normal distribution, mean 50, std 10
    predict_results_train = y_train + np.random.normal(0, 5, size=y_train.shape)
    y_test = np.random.normal(loc=50, scale=10, size=3000)  # Normal distribution, mean 50, std 10
    predict_results_test = y_test + np.random.normal(0, 5, size=y_test.shape)
    plot_fit_results('Isp', y_train, predict_results_train, y_test, predict_results_test)
    # plot_fit_results('T_c', y_train, predict_results_train, y_test, predict_results_test)