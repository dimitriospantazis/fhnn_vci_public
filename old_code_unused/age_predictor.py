from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import IsolationForest
from nn_regressor import Neural_Network_Regressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.data_utils import THICK_INDEX, MYELIN_INDEX
from utils.data_utils import min_max_normalize
from utils.data_utils import remove_self_loops
from utils.constants_utils import FIVE_PERCENT_THRESHOLD
# from umap import UMAP
import matplotlib.pyplot as plt
from visualization import to_poincare
import torch
from hyperbolic_clustering.utils.utils import poincare_dist
import numpy as np 
import networkx as nx
from typing import List
from typing import Tuple    
import time
from tqdm import tqdm
import numpy as np
import os 
import pickle
from utils.constants_utils import NUM_ROIS, NUM_SBJS, NUM_MEG_COLE_ROIS
BAR_WIDTH = 0.35
age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
meg_age_labels = np.load(os.path.join("data", "meg", "MEG_age_labels_180.npy"))

class Age_Model:
    def __init__(self, date : str, log_num : str, projection_type: str ="HR", dataset: str ="Cam-CAN"):
        log_path = os.path.join("logs", "lp", date, log_num)
        print(f"Using Log Path : {log_path}")
        self.embeddings_dir = os.path.join(log_path, 'embeddings')
        projection_type = projection_type.replace("-", "").upper()
        self.projection_type = projection_type
        self.val_indices = self.get_split_indices("val")
        self.test_indices = self.get_split_indices("test")
        self.train_indices = self.get_split_indices("train")
        # TODO: RESTOR TRAIN INDICES TO THIS vvv
        # self.train_indices = [i for i in range(NUM_SBJS) if i not in self.val_indices and i not in self.test_indices]
        # print(self.train_indices, "THESE ARE THE TRAIN INDICES")
        self.dataset = dataset
        if dataset not in ["Cam-CAN", "MEG"]:
            raise ValueError("Dataset must be either Cam-CAN or MEG")
        self.num_rois = NUM_ROIS if self.dataset == "Cam-CAN" else NUM_MEG_COLE_ROIS

    def project_embeddings(self, embeddings_list) -> np.ndarray:
        if self.projection_type == "DISTANCE_MATRIX_CONCATENATION":
            sbj_hyp_dist_matrices_flattened = np.zeros((len(embeddings_list), (NUM_ROIS * (NUM_ROIS - 1)) // 2))
            def inner_product(u, v):
                return -u[0]*v[0] + np.dot(u[1:], v[1:]) 
            def get_hyperbolic_distance(embeddings_i, embeddings_j):
                # origin = np.array([1, 0, 0]) # .to(self.args.device)
                return np.nan_to_num(np.arccosh(-1 * inner_product(embeddings_i, embeddings_j)))
            def get_hyperbolic_radius(embeddings):
                origin = np.array([1, 0, 0]) # .to(self.args.device)
                return [np.arccosh(-1 * inner_product(origin, coord)) for coord in embeddings]
            
            for sbj_index in range(len(embeddings_list)):
                embeddings = embeddings_list[sbj_index]
                hyp_dist_matrix_flattened = []

                for i in range(NUM_ROIS):
                    for j in range(i + 1, NUM_ROIS):
                        hyp_dist = get_hyperbolic_distance(embeddings[i], embeddings[j])
                        hyp_dist_matrix_flattened.append(hyp_dist)
                hyp_dist_matrix_flattened = np.array(hyp_dist_matrix_flattened)
                
                include_hyperbolic_radii = True
                print(f"Include Hyperbolic Radii : {include_hyperbolic_radii}")
                if include_hyperbolic_radii:
                    hyperbolic_radii = np.array([get_hyperbolic_radius(embeddings[i]) for i in range(NUM_ROIS)])
                    hyp_dist_matrix_flattened = np.hstack([hyp_dist_matrix_flattened, hyperbolic_radii])
                sbj_hyp_dist_matrices_flattened[sbj_index] = hyp_dist_matrix_flattened
            return sbj_hyp_dist_matrices_flattened

        if self.projection_type == "FLATTEN":
            return np.array([np.array(embeddings).flatten() for embeddings in embeddings_list])
        projection_function = self.get_projection_function()
        # scaled_embeddings = self.scale_embeddings(embeddings_list)
        # projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(scaled_embeddings)]
        projected_embeddings = [projection_function(embeddings) for embeddings in embeddings_list]
        np_projected_embeddings = np.array(projected_embeddings)
        if self.projection_type != "HR": 
            projected_embeddings = np_projected_embeddings.reshape((len(np_projected_embeddings), self.num_rois))
        return projected_embeddings
    
    def detect_outliers(self, projected_embeddings) -> List[int]:
        """
        Detect Outliers, returns a list of 1's and -1's, 1 if inlier, -1 if outlier of size len(projected_embeddings)
        """
        detector_model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter

        is_inlier_or_outlier_list = detector_model.fit_predict(projected_embeddings)
        num_outliers = np.count_nonzero(is_inlier_or_outlier_list == -1)
        print(f"Detected {num_outliers} Outliers")
        return is_inlier_or_outlier_list

    # TODO: Figure out if should scale before or after projection
    def scale_embeddings(self, embeddings_list) -> np.ndarray:
        print("Scaling Embeddings :")
        scaler = StandardScaler()
        scaled_embeddings = [scaler.fit_transform(embeddings) for embeddings in embeddings_list]
        np_scaled_embeddings = np.array(scaled_embeddings)
        scaled_embeddings = np_scaled_embeddings.reshape((len(np_scaled_embeddings), self.num_rois))
        return scaled_embeddings
    def get_projection_function(self):
        projection_function = lambda x : x
        if self.projection_type == "HR": 
            
            def inner_product(u, v):
                return -u[0]*v[0] + np.dot(u[1:], v[1:]) 
            def get_hyperbolic_radius(embeddings):
                origin = np.array([1, 0, 0]) # .to(self.args.device)
                return [np.arccosh(-1 * inner_product(origin, coord)) for coord in embeddings]
            
            def get_poincare_radius(embeddings):
                poincare_origin = torch.Tensor([0, 0]) 
                torch_embeddings = torch.from_numpy(embeddings)
                c = 1.0
                poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
                return [poincare_dist(poincare_embedding, poincare_origin) for poincare_embedding in poincare_embeddings]
            # def get_squared_radius(embeddings):
            #     return [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            # projection_function = get_poincare_radius
            projection_function = get_hyperbolic_radius

        elif self.projection_type == "TSNE": projection_function = TSNE(n_components=1, init='random', perplexity=3).fit_transform
        elif self.projection_type == "SVD": projection_function = TruncatedSVD(n_components=1).fit_transform
        elif self.projection_type == "PCA": projection_function = PCA(n_components=1).fit_transform
        elif self.projection_type == "ISOMAP": projection_function = Isomap(n_components=1).fit_transform
        elif self.projection_type == "UMAP": projection_function = UMAP(n_components=1).fit_transform
        else: raise AssertionError(f"Invalid Projection Type : {self.projection_type}!")
        # Other possibilities: MDS, LLE, Laplacian Eigenmaps, etc.
        return projection_function
            
    def get_split_indices(self, split_str):
        split_indices = []
        for split_embeddings_dir in os.listdir(self.embeddings_dir):
            if split_str not in split_embeddings_dir: continue
            _, _, split_index_str = split_embeddings_dir.split("_")
            split_index, _ = split_index_str.split(".")
            split_index = int(split_index)
            split_indices.append(split_index)
        return sorted(split_indices)
        # Sorting the indices should not affect age prediction results... must investigate further...
        # return split_indices

class Age_Predictor(Age_Model):
    def __init__(self, date : str, log_num : str, type_of_regression: str, projection_type: str="HR", architecture: str="FHNN", dataset: str="Cam-CAN", 
                 alpha=100, use_viz=False, seed = 9999):
        """
        1. Data MEG MRI fMRI
            Access PLV Subject Matrices 
        
        2. Get Adjacency Matrix
            Define Threshold
            Binarize Matrix
        
        3. Create Brain Graph
            Training Set    
            Validation Set 
            Test Set
        
        4. Create HGCNN Embeddings
            Visualize Embeddings by plotting in Poincare Disk --> Drew Wilimitis Code
        
        5. Ridge Regression
        
        6. Evaluate Regression Model: MSE
        
        7. Visualize Predicted Age vs. Actual Age
        """
        super().__init__(date, log_num, projection_type, dataset)
        type_of_regression = type_of_regression.lower()
        self.architecture = architecture
        self.dataset = dataset
        self.use_viz = use_viz
        np.random.seed(seed)
        # self.use_connectivity = False
        if type_of_regression == "linear":
            self.regressor_model = LinearRegression()
        elif type_of_regression == "ridge":
            print("Alpha Parameter :", alpha)
            self.regressor_model = Ridge(alpha = alpha)
        elif type_of_regression == "polynomial":
            raise AssertionError("Polynomial Regression not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.regressor_model = LinearRegression()
        elif type_of_regression == "hyperbolic":
            raise AssertionError("Hyperbolic Regression not implemented yet!")
            self.regressor_model = HyperbolicCentroidRegression()
        elif type_of_regression == "random_forest":
            # self.regressor_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.regressor_model = RandomForestRegressor(n_estimators=50, random_state=42)
            # self.regressor_model = RandomForestRegressor(n_estimators=500, random_state=42)
        elif type_of_regression == "neural_network":
            use_thickness_myelination = False
            use_thickness_myelination_plv = False

            if use_thickness_myelination: input_size = hidden_size = NUM_ROIS * 2
            elif use_thickness_myelination_plv: input_size = hidden_size = NUM_ROIS * 3 
            else: input_size = hidden_size = NUM_ROIS
            num_modalities = 1
            input_size = hidden_size = NUM_ROIS * num_modalities
            # hidden_size = 200
            output_size = 1
            lr = 0.001
            # lr = 0.0005
            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, learning_rate=lr)
        elif type_of_regression == "huber":
            # epsilon = 1.35
            epsilon = 1.0
            max_iter = 100
            self.regressor_model = HuberRegressor(epsilon=epsilon, max_iter=max_iter)
        elif type_of_regression == "linear_svr":
            self.regressor_model = LinearSVR()
        elif type_of_regression == "rbf_svr":
            C = 1e3
            gamma = 10
            self.regressor_model = SVR(kernel='rbf', C=C, gamma=gamma)
        elif type_of_regression == "gaussian_process":
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e2, 1e2))
            self.regressor_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        elif type_of_regression == "decision_tree":
            self.regressor_model = DecisionTreeRegressor(max_depth=5)
        elif type_of_regression == 'k_neighbors':
            self.regressor_model = KNeighborsRegressor(n_neighbors=5)
        elif type_of_regression == 'ada_boost':
            base_regressor = DecisionTreeRegressor(max_depth=5)
            # NOTE: Random state is currently fixed to 42
            self.regressor_model = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)
            # self.regressor_model = AdaBoostRegressor(base_regressor, n_estimators=10, random_state=42)
        else:
            raise AssertionError(f"Invalid Regression type : {type_of_regression}!")
        if dataset == "Cam-CAN":
            self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        elif dataset == "MEG":
            self.train_age_labels = [meg_age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [meg_age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [meg_age_labels[test_index] for test_index in self.test_indices]
        else:
            raise AssertionError(f"Invalid Dataset : {dataset}!")
        self.model_str = "Linear" if type(self.regressor_model) == LinearRegression \
            else "Ridge" if type(self.regressor_model) == Ridge \
            else "Random Forest" if type(self.regressor_model) == RandomForestRegressor \
            else "Feed-Forward NN" if type(self.regressor_model) == Neural_Network_Regressor \
            else "Huber Regressor" if type(self.regressor_model) == HuberRegressor \
            else "Linear Support Vector Regressor" if type(self.regressor_model) == LinearSVR \
            else "RBF Support Vector Regressor" if type(self.regressor_model) == SVR \
            else "Gaussian Process Regressor" if type(self.regressor_model) == GaussianProcessRegressor \
            else "Decision Tree Regressor" if type(self.regressor_model) == DecisionTreeRegressor \
            else 'K Neighbors Regressor' if type(self.regressor_model) == KNeighborsRegressor \
            else 'Ada Boost Regressor' if type(self.regressor_model) == AdaBoostRegressor \
            else "Unknown"
        
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        if self.use_viz and \
            type(self.regressor_model) in [LinearRegression, Ridge, Neural_Network_Regressor, RandomForestRegressor, HuberRegressor]:
            
            if self.projection_type != "DISTANCE_MATRIX_CONCATENATION": self.visualize_model_parameters(use_jet = False)
        # TODO: Uncomment this
        # self.plot_difference_between_predicted_and_labels(predicted_ages)
        if self.use_viz:
            self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
            self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def visualize_model_parameters(self, use_jet=False):
        # plt.figure(figsize = (10, 10))

        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        # x = np.arange(self.num_rois)
        # NUM_COORDINATES = 3
        # if self.projection_type == "FLATTEN": x = np.arange(NUM_COORDINATES * self.num_rois)
        
        cmap = plt.cm.jet        
        
        if type(self.regressor_model) == RandomForestRegressor:
            x = np.arange(len(self.regressor_model.feature_importances_))
            plt.bar(x, self.regressor_model.feature_importances_, color=cmap(x / len(x)))
            return 
        elif type(self.regressor_model) == Neural_Network_Regressor:
            print(f"Model Parameters : {[tensor.shape for tensor in self.regressor_model.parameters()]}")
            row_sums = [torch.sum(tensor, dim=0).detach().numpy() for tensor in self.regressor_model.parameters()]
            print("ROW SUMS : ", row_sums)
            x = np.arange(len(row_sums[0]))
            plt.bar(x, row_sums[0], color=cmap(x / len(x)))
            return
        
            # plt.imshow(self.regressor_model.parameters())
        if use_jet:
            x = np.arange(len(self.regressor_model.coef_))
            plt.bar(x, self.regressor_model.coef_, color=cmap(x / len(x)))
        else:
            x = np.arange(len(self.regressor_model.coef_))
            plt.bar(x, self.regressor_model.coef_)

    def plot_age_labels_vs_predicted_ages(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # Set width of each bar

        # Plotting the barplots
        
        plt.bar(x - BAR_WIDTH/2, self.test_age_labels, BAR_WIDTH, label='Age Label')
        plt.bar(x + BAR_WIDTH/2, predicted_ages, BAR_WIDTH, label='Predicted Age')

        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type}')
        plt.legend()
        
        # Show the plot
        plt.show()
    
    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        # x = np.arange(len(self.test_indices + self.val_indices))
        # plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        

    def plot_difference_between_predicted_and_labels(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        
        plt.bar(x - BAR_WIDTH/2, predicted_ages - self.test_age_labels, BAR_WIDTH, label='Predicted Age - Age Label')
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # ax.set_xticklabels(self.test_indices, rotation=90)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Difference Plot')
        plt.legend()

        plt.show()

    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation Coefficient, and R^2 between Predicted Ages and Actual Ages
        """
        
        test_embeddings_list = []
        val_embeddings_list = []
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        # TODO: Change back to only test_embeddings_list
        # test_embeddings_list += val_embeddings_list
        # print("Projecting Test Embeddings :")
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        
        # print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        # test_embeddings_list = [test_embeddings for index, test_embeddings in enumerate(test_embeddings_list) \
        #                             if is_inlier_or_outlier_list[index] == 1]
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) != Neural_Network_Regressor:
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)
            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        # print("Age Labels: ", self.test_age_labels)
        # print("Predicted Ages: ", predicted_ages)
        
        # TODO: RESTORE TO ONLY TEST AGE LABELS
        # return predicted_ages, \
                # mean_absolute_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # mean_squared_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # np.corrcoef(predicted_ages, self.test_age_labels + self.val_age_labels)[0, 1], \
                # r2_score(predicted_ages, self.test_age_labels + self.val_age_labels)
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                age_label = age_labels[train_index]
                # meg_age_label = meg_age_labels[train_index]

                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                # TODO: Matrix to Label seems inefficient
                embeddings_to_labels[tuple(embeddings)] = age_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        
        # train_embeddings_list += val_embeddings_list
        # train_embeddings_list = val_embeddings_list
        # TODO: Change back to train + val
        # train_embeddings_list += test_embeddings_list
        # print("Projecting Train Embeddings :")
        projected_embeddings = self.project_embeddings(train_embeddings_list)
        # print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        # train_embeddings_list = [train_embeddings for index, train_embeddings in enumerate(train_embeddings_list) \
        #                          if is_inlier_or_outlier_list[index] == 1]
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) != Neural_Network_Regressor:
            # Projection Mapping from 3D to 1D
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
            # TODO: Change back to train + val
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels + self.test_age_labels)
        elif type(self.regressor_model) == Neural_Network_Regressor:

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            # age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels + self.val_age_labels)) \
            #     .clone() \
            #     .detach() \
            #     .to(dtype=torch.float32) \
            #     .squeeze()
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def predict_age(self, embeddings) -> float:
        """
        Predict age from embeddings, no label difference calculation
        """
        if type(self.regressor_model) == LinearRegression:
            radii_sqs = [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            predicted_age = self.regressor_model.predict(radii_sqs)
        return predicted_age

    def mse_loss(self, predicted_ages_with_age_labels) -> float:    
        """
        Return mean-squared errors between predicted and actual ages
        
        """
        # mse_loss = sum((predicted_age - age_label) ** 2 for predicted_age, age_label 
        #     in predicted_ages_with_age_labels) / len(predicted_ages_with_age_labels)
        predicted_ages = [pred_label[0] for pred_label in predicted_ages_with_age_labels]
        respective_age_labels = [pred_label[1] for pred_label in predicted_ages_with_age_labels]
        mse_loss = mean_squared_error(predicted_ages, respective_age_labels)
        return mse_loss



class Age_Predictor_for_Node_Level_Measures(Age_Predictor):
    def __init__(self, 
                 measure_str : str,
                 date : str, 
                 log_num : str, 
                 type_of_regression: str, 
                 projection_type: str="HR", 
                 architecture: str="FHNN", 
                 dataset: str="Cam-CAN", 
                 alpha=100,
                 use_viz=True,
                 seed=9999):

        super().__init__(date, 
                    log_num, 
                    type_of_regression, 
                    projection_type, 
                    architecture, 
                    dataset, 
                    alpha)
        self.seed = seed
        np.random.seed(self.seed)
        self.train_indices, self.val_indices, self.test_indices = self.get_dataset_split_indices()
        # self.train_indices, self.val_indices, self.test_indices = sorted(self.train_indices), sorted(self.val_indices), sorted(self.test_indices)
        self.train_indices.sort()
        self.val_indices.sort()
        self.test_indices.sort()
        self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
        self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
        self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        self.measure_str = measure_str
        self.use_viz = use_viz
        self.use_connectivity = False
        
        print(f"Use PLV Connectivity Matrices : {self.use_connectivity}")
        
        if self.measure_str in ['myelination', 'thickness', 'thickness_myelination']:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            thick_features = thicks_myelins_tensor[THICK_INDEX]
            thick_features = np.reshape(thick_features, (NUM_SBJS, NUM_ROIS))
            myelin_features = thicks_myelins_tensor[MYELIN_INDEX] 
            myelin_features = np.reshape(myelin_features, (NUM_SBJS, NUM_ROIS))
            use_min_max_scaler = False
            
            if use_min_max_scaler:
                thick_features = min_max_normalize(thick_features)
                myelin_features = min_max_normalize(myelin_features)    
            if self.measure_str == "myelination": self.structural_measures = myelin_features
            if self.measure_str == "thickness": self.structural_measures = thick_features
            if self.measure_str == "thickness_myelination": self.structural_measures = np.hstack((thick_features, myelin_features))
        if self.measure_str == "plv":
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            use_partial_thresholding = True
            print(f"Using Partial Thresholding : {use_partial_thresholding}")
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]

                if use_partial_thresholding:
                    plv_matrix[plv_matrix < FIVE_PERCENT_THRESHOLD] = 0
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector
                
            use_min_max_scaler = False
            if use_min_max_scaler: sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            self.structural_measures = sbj_plv_vectors
        if self.measure_str in ["thickness_myelination_plv", "thickness_plv", "myelination_plv"]:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            thick_features = thicks_myelins_tensor[THICK_INDEX]
            thick_features = np.reshape(thick_features, (NUM_SBJS, NUM_ROIS))
            myelin_features = thicks_myelins_tensor[MYELIN_INDEX] 
            myelin_features = np.reshape(myelin_features, (NUM_SBJS, NUM_ROIS))
            use_min_max_scaler = False
            
            if use_min_max_scaler:
                thick_features = min_max_normalize(thick_features)
                myelin_features = min_max_normalize(myelin_features)    
                sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            if self.measure_str == "thickness_myelination_plv": 
                self.structural_measures = np.hstack((thick_features, myelin_features, sbj_plv_vectors))
            elif self.measure_str == "myelination_plv": 
                self.structural_measures = np.hstack((myelin_features, sbj_plv_vectors))
            elif self.measure_str == "thickness_plv": 
                self.structural_measures = np.hstack((thick_features, sbj_plv_vectors))
            else: raise AssertionError("Invalid access to measure string conditional")

        if self.use_connectivity:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector
            use_min_max_scaler = True
            if use_min_max_scaler: sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            self.plv_vectors = sbj_plv_vectors

        if self.model_str == "Feed-Forward NN":
            num_modalities = 1
            if self.measure_str in ['thickness', 'myelination', 'plv']: num_modalities = 1
            if self.measure_str in ['thickness_myelination', 'myelination_plv', 'thickness_plv']: num_modalities = 2
            if self.measure_str in ['thickness_myelination_plv']: num_modalities = 3
            input_size = hidden_size = NUM_ROIS * num_modalities
            output_size = 1
            lr = 0.001
            if self.measure_str in ['plv', 'thickness_plv', 'myelination_plv', 'thickness_myelination_plv']:
                input_size = hidden_size = (NUM_ROIS * (NUM_ROIS + 1)) // 2 + (num_modalities - 1) * NUM_ROIS

            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, lr)
            
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        print("THIS IS USE VIZ", self.use_viz)
        if self.use_viz:
            if type(self.regressor_model) in [LinearRegression, Ridge, Neural_Network_Regressor, RandomForestRegressor, HuberRegressor]:
                self.visualize_model_parameters(use_jet = False)
        # self.plot_difference_between_predicted_and_labels(predicted_ages)
        if self.use_viz:
            self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
            self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation, R^2 Coefficient between Predicted Ages and Actual Ages
        """
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        # with open('node_level_measures_dict.pkl', 'rb') as file:
        #     node_level_measures_dict = pickle.load(file)

        if self.measure_str in ["thickness", "myelination", "plv", 
                                "thickness_myelination", "myelination_plv", "thickness_plv",
                                "thickness_myelination_plv"]:
            measures = self.structural_measures
        else:
            # with open('node_level_and_hyper_measures.pkl', 'rb') as file:
                # node_level_measures_dict = pickle.load(file)
            with open('node_level_measures_partial_thresholding.pkl', 'rb') as file:
                node_level_measures_dict = pickle.load(file)
            measures = node_level_measures_dict[self.measure_str]
        
        # train_embeddings_list = [measures[train_index] for train_index in self.train_indices]
        # val_embeddings_list = [measures[val_index] for val_index in self.val_indices]
        if self.use_connectivity:
            measures = np.array([measures[sbj_index] for sbj_index in range(NUM_SBJS)])
            measures = np.hstack([self.plv_vectors, measures])    
        test_embeddings_list = [measures[test_index] for test_index in self.test_indices]
        
        projected_embeddings = test_embeddings_list
            
        # print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        # train_embeddings_list += val_embeddings_list
        # TODO: Change back to only test_embeddings_list
        # test_embeddings_list += val_embeddings_list
        # Detect and prune out outliers from regression
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)

        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) != Neural_Network_Regressor:            
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)

            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        # print("Age Labels: ", self.test_age_labels)
        # print("Predicted Ages: ", predicted_ages)
        
        # TODO: RESTORE TO ONLY TEST AGE LABELS
        # return predicted_ages, \
                # mean_absolute_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # mean_squared_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # np.corrcoef(predicted_ages, self.test_age_labels + self.val_age_labels)[0, 1], \
                # r2_score(predicted_ages, self.test_age_labels + self.val_age_labels)
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        # val_embeddings_list = []
        # test_embeddings_list = []
        # with open('node_level_measures_dict.pkl', 'rb') as file:
        #     node_level_measures_dict = pickle.load(file)
        if self.measure_str in ["thickness", "myelination", "plv", 
                                "thickness_myelination", "myelination_plv", "thickness_plv",
                                "thickness_myelination_plv"]:
            measures = self.structural_measures

        else:
            # with open('node_level_and_hyper_measures.pkl', 'rb') as file:
                # node_level_measures_dict = pickle.load(file)
            with open('node_level_measures_partial_thresholding.pkl', 'rb') as file:
                node_level_measures_dict = pickle.load(file)
            measures = node_level_measures_dict[self.measure_str]
        if self.use_connectivity:
            
            measures = np.array([measures[sbj_index] for sbj_index in range(NUM_SBJS)])
            
            measures = np.hstack([self.plv_vectors, measures])        
        train_embeddings_list = [measures[train_index] for train_index in self.train_indices]
        # val_embeddings_list = [measures[val_index] for val_index in self.val_indices]
        # test_embeddings_list = [measures[test_index] for test_index in self.test_indices]

        projected_embeddings = train_embeddings_list
        # print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)

        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]

        # train_embeddings_list += val_embeddings_list
        # train_embeddings_list = val_embeddings_list
        # TODO: Change back to train + val
        # train_embeddings_list += test_embeddings_list
        if type(self.regressor_model) != Neural_Network_Regressor:
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
            # TODO: Change back to train + val
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels + self.test_age_labels)
        elif type(self.regressor_model) == Neural_Network_Regressor:
            
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            # age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels + self.val_age_labels)) \
            #     .clone() \
            #     .detach() \
            #     .to(dtype=torch.float32) \
            # #     .squeeze()
            
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            # TODO: Change back to train + val
            # age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels + self.val_age_labels + self.test_age_labels)) \
            #     .clone() \
            #     .detach() \
            #     .to(dtype=torch.float32) \
            #     .squeeze()
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model using {self.measure_str} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        # x = np.arange(len(self.test_indices + self.val_indices))
        # plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model using {self.measure_str} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        


    # @staticmethod    
    def get_dataset_split_indices(self) -> List[List[int]]:
        
        train_split_indices, val_split_indices, test_split_indices = [], [], []
        import logging
        nth_index = 0
        nth_str = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"][nth_index]
        num_sbjs = NUM_SBJS
        TRAIN_SPLIT = 0.7
        VAL_SPLIT = 0.2
        TEST_SPLIT = 0.1
        logging.info(f"Using Only {nth_str} {num_sbjs} Subjects")
        train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
        test_num = num_sbjs - train_num - val_num 

        seen = set()
        lower_bound = num_sbjs * nth_index
        upper_bound = lower_bound + num_sbjs
        # upper_bound = 587 # [490, 587) -> 97
        logging.info("Using 100% of subject indices as training set")
        for split_indices, split_num in zip([train_split_indices, val_split_indices, test_split_indices],
                                        [train_num, val_num, test_num]):
            num_indices = 0
            while num_indices < split_num:
                # index = np.random.randint(0, num_sbjs) # Do not forget filtering! (subtract 5)        
                index = np.random.randint(lower_bound, upper_bound) # Do not forget filtering! (subtract 5)        
                if index in seen: continue
                split_indices.append(index)
                num_indices += 1
                seen.add(index)
        assert not set(train_split_indices).intersection(val_split_indices), "Train and Val Sets should not overlap"
        assert not set(train_split_indices).intersection(test_split_indices), "Train and Test Sets should not overlap"
        assert not set(val_split_indices).intersection(test_split_indices), "Val and Test Sets should not overlap"
            
        return train_split_indices, val_split_indices, test_split_indices
    
class Structural_Age_Predictor(Age_Predictor):
    def __init__(self, 
                 measure_str : str,
                 date : str, 
                 log_num : str, 
                 type_of_regression: str, 
                 projection_type: str="HR", 
                 architecture: str="FHNN", 
                 dataset: str="Cam-CAN", 
                 alpha=100,
                 use_viz=True,
                 seed=9999):

        super().__init__(date, 
                    log_num, 
                    type_of_regression, 
                    projection_type, 
                    architecture, 
                    dataset, 
                    alpha,
                    use_viz=use_viz)
        self.seed = seed
        np.random.seed(self.seed)
        # self.train_indices, self.val_indices, self.test_indices = sorted(self.train_indices), sorted(self.val_indices), sorted(self.test_indices)
        self.train_indices.sort()
        self.val_indices.sort()
        self.test_indices.sort()
        self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
        self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
        self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        self.measure_str = measure_str
        use_min_max_scaler = True
        self.use_min_max_scaler = use_min_max_scaler
        self.use_connectivity = False
        print(f"Use PLV Connectivity Matrices : {self.use_connectivity}")
        print(f"Use Min-Max Scaler: {use_min_max_scaler}")

        if self.measure_str in ['myelination', 'thickness', 'thickness_myelination']:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            thick_features = thicks_myelins_tensor[THICK_INDEX]
            thick_features = np.reshape(thick_features, (NUM_SBJS, NUM_ROIS))
            myelin_features = thicks_myelins_tensor[MYELIN_INDEX] 
            myelin_features = np.reshape(myelin_features, (NUM_SBJS, NUM_ROIS))
            
            if use_min_max_scaler:
                thick_features = min_max_normalize(thick_features)
                myelin_features = min_max_normalize(myelin_features)    
            if self.measure_str == "myelination": self.structural_measures = myelin_features
            if self.measure_str == "thickness": self.structural_measures = thick_features
            if self.measure_str == "thickness_myelination": self.structural_measures = np.hstack((thick_features, myelin_features))
        elif self.measure_str == "plv":
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]
                use_partial_thresholding = True
                print(f"Using Partial Thresholding : {use_partial_thresholding}")
                if use_partial_thresholding:
                    plv_matrix[plv_matrix < FIVE_PERCENT_THRESHOLD] = 0
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector

            if use_min_max_scaler: sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            self.structural_measures = sbj_plv_vectors
        elif self.measure_str in ["thickness_myelination_plv", "thickness_plv", "myelination_plv"]:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector
            thicks_myelins_tensor = np.load(os.path.join(data_path, "cam_can_thicks_myelins_tensor_592_filtered.npy")) # 592 as well  
            thick_features = thicks_myelins_tensor[THICK_INDEX]
            thick_features = np.reshape(thick_features, (NUM_SBJS, NUM_ROIS))
            myelin_features = thicks_myelins_tensor[MYELIN_INDEX] 
            myelin_features = np.reshape(myelin_features, (NUM_SBJS, NUM_ROIS))
            
            if use_min_max_scaler:
                thick_features = min_max_normalize(thick_features)
                myelin_features = min_max_normalize(myelin_features)    
                sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            if self.measure_str == "thickness_myelination_plv": 
                self.structural_measures = np.hstack((sbj_plv_vectors, thick_features, myelin_features))
            elif self.measure_str == "myelination_plv": 
                self.structural_measures = np.hstack((myelin_features, sbj_plv_vectors))
            elif self.measure_str == "thickness_plv": 
                self.structural_measures = np.hstack((thick_features, sbj_plv_vectors))
            else: raise AssertionError("Invalid access to measure string conditional")
        else:
            print("Partial Thresholding Graph Measures:")
            # with open('node_level_and_hyper_measures.pkl', 'rb') as file:
            # # with open('node_level_measures_dict.pkl', 'rb') as file:
            #     node_level_measures_dict = pickle.load(file)
            with open('node_level_measures_partial_thresholding.pkl', 'rb') as file:
                node_level_measures_dict = pickle.load(file)
            self.structural_measures = node_level_measures_dict[self.measure_str]
            self.structural_measures = np.array([self.structural_measures[sbj_index] for sbj_index in range(NUM_SBJS)])
            if use_min_max_scaler:
                self.structural_measures = min_max_normalize(self.structural_measures)
                # myelin_features = min_max_normalize(myelin_features)  
        if self.use_connectivity:
            data_path = os.path.join(os.getcwd(), "data", "cam_can_multiple")
            plv_tensor = np.load(os.path.join(data_path, "plv_tensor_592_sbj_filtered.npy"))            
            plv_tensor = remove_self_loops(plv_tensor)
            sbj_plv_vectors = np.zeros((NUM_SBJS, (NUM_ROIS * (NUM_ROIS + 1)) // 2))
            for i in range(NUM_SBJS):
                plv_matrix = plv_tensor[i]
                # Convert the symmetric matrix to a 1D vector
                plv_vector = plv_matrix[np.triu_indices(NUM_ROIS)]
                sbj_plv_vectors[i] = plv_vector
            use_min_max_scaler = True
            if use_min_max_scaler: sbj_plv_vectors = min_max_normalize(sbj_plv_vectors)    
            self.plv_vectors = sbj_plv_vectors
        if self.model_str == "Feed-Forward NN":
            print(f"THE MODEL STRING IS {self.model_str}")
            num_modalities = 2
            if self.measure_str in ['thickness', 'myelination', 'plv']: num_modalities = 2
            if self.measure_str in ['thickness_myelination', 'myelination_plv', 'thickness_plv']: num_modalities = 3
            if self.measure_str in ['thickness_myelination_plv']: num_modalities = 4
            input_size = hidden_size = NUM_ROIS * num_modalities
            if self.measure_str in ['plv', 'thickness_plv', 'myelination_plv', 'thickness_myelination_plv']:
                input_size = hidden_size = (NUM_ROIS * (NUM_ROIS + 1)) // 2 + (num_modalities - 1) * NUM_ROIS
            output_size = 1
            lr = 0.001
            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, lr)
    
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        if self.use_viz and \
            type(self.regressor_model) in [LinearRegression, Ridge, Neural_Network_Regressor, RandomForestRegressor, HuberRegressor]:
            pass
            # self.visualize_model_parameters(use_jet = False)
        # TODO: Uncomment this
        # self.plot_difference_between_predicted_and_labels(predicted_ages)
        if self.use_viz:
            self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
            self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} using {self.measure_str} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} using {self.measure_str} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} using {self.measure_str} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} using {self.measure_str} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def visualize_model_parameters(self, use_jet=False):
        # plt.figure(figsize = (10, 10))
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} using {self.measure_str} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        x = np.arange(self.num_rois)
        cmap = plt.cm.jet        
        
        if type(self.regressor_model) == RandomForestRegressor:
            x = np.arange(len(self.regressor_model.feature_importances_))
            plt.bar(x, self.regressor_model.feature_importances_, color=cmap(x / len(x)))
            return 
        elif type(self.regressor_model) == Neural_Network_Regressor:
            print(f"Model Parameters : {[tensor.shape for tensor in self.regressor_model.parameters()]}")
            row_sums = [torch.sum(tensor, dim=0).detach().numpy() for tensor in self.regressor_model.parameters()]
            print("ROW SUMS : ", row_sums)
            x = np.arange(len(row_sums[0]))
            plt.bar(x, row_sums[0], color=cmap(x / len(x)))
            return
        
            # plt.imshow(self.regressor_model.parameters())
        if use_jet:
            x = np.arange(len(self.regressor_model.coef_))
            plt.bar(x, self.regressor_model.coef_, color=cmap(x / len(x)))
        else:
            x = np.arange(len(self.regressor_model.coef_))
            plt.bar(x, self.regressor_model.coef_)

    def plot_age_labels_vs_predicted_ages(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # Set width of each bar

        # Plotting the barplots
        
        plt.bar(x - BAR_WIDTH/2, self.test_age_labels, BAR_WIDTH, label='Age Label')
        plt.bar(x + BAR_WIDTH/2, predicted_ages, BAR_WIDTH, label='Predicted Age')

        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type}')
        plt.legend()
        
        # Show the plot
        plt.show()
    
    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        # x = np.arange(len(self.test_indices + self.val_indices))
        # plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        

    def plot_difference_between_predicted_and_labels(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        
        plt.bar(x - BAR_WIDTH/2, predicted_ages - self.test_age_labels, BAR_WIDTH, label='Predicted Age - Age Label')
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # ax.set_xticklabels(self.test_indices, rotation=90)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Difference Plot')
        plt.legend()

        plt.show()

    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation Coefficient, and R^2 between Predicted Ages and Actual Ages
        """
        
        test_embeddings_list = []
        val_embeddings_list = []
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        # for val_index in self.val_indices:
        #     val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
        #     val_embeddings_list.append(val_embeddings)
        # COMBINE STRUCTURAL AND HYPERBOLIC EMBEDDINGS 
        # val_embeddings_list = [self.structural_measures[val_index] + val_embeddings_list[val_index] for val_index in self.val_indices]
        # print("Projecting Test Embeddings :")
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        if self.use_min_max_scaler:
            projected_embeddings = min_max_normalize(projected_embeddings)
        
        
        if self.use_connectivity:
            projected_embeddings = [np.hstack((self.plv_vectors[test_index], 
                                               self.structural_measures[test_index],  
                                               projected_embeddings[index])) for index, test_index in enumerate(self.test_indices)]
               
        else:
            projected_embeddings = [np.hstack((self.structural_measures[test_index], projected_embeddings[index])) for index, test_index in enumerate(self.test_indices)]
        # print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)

        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        # test_embeddings_list = [test_embeddings for index, test_embeddings in enumerate(test_embeddings_list) \
        #                             if is_inlier_or_outlier_list[index] == 1]
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) != Neural_Network_Regressor:
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)
            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        # print("Age Labels: ", self.test_age_labels)
        # print("Predicted Ages: ", predicted_ages)
        
        # TODO: RESTORE TO ONLY TEST AGE LABELS
        # return predicted_ages, \
                # mean_absolute_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # mean_squared_error(predicted_ages, self.test_age_labels + self.val_age_labels), \
                # np.corrcoef(predicted_ages, self.test_age_labels + self.val_age_labels)[0, 1], \
                # r2_score(predicted_ages, self.test_age_labels + self.val_age_labels)
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                age_label = age_labels[train_index]
                # meg_age_label = meg_age_labels[train_index]

                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                # TODO: Matrix to Label seems inefficient
                embeddings_to_labels[tuple(embeddings)] = age_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels
        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        # for val_index in self.val_indices:
        #     val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
        #     val_embeddings_list.append(val_embeddings)
        # for test_index in self.test_indices:
        #     test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
        #     test_embeddings_list.append(test_embeddings)
        
        # COMBINE STRUCTURAL AND HYPERBOLIC EMBEDDINGS
        # train_embeddings_list = [self.structural_measures[train_index] + train_embeddings_list[train_index] for train_index in self.train_indices]
        # val_embeddings_list = [measures[val_index] + val_embeddings_list[val_index] for val_index in self.val_indices]
        # test_embeddings_list = [measures[test_index] + test_embeddings_list[test_index] for test_index in self.test_indices]

        # print("Projecting Train Embeddings :")
        projected_embeddings = self.project_embeddings(train_embeddings_list)
        if self.use_min_max_scaler:
            projected_embeddings = min_max_normalize(projected_embeddings)
        # projected_embeddings = [np.hstack((self.structural_measures[train_index], projected_embeddings[index])) for index, train_index in enumerate(self.train_indices)]
        # NOTE: Be very careful with keeping train_index and index relationship intact so as to not mix data per subject!
        if self.use_connectivity:
            projected_embeddings = [np.hstack((self.plv_vectors[train_index], 
                                               self.structural_measures[train_index],  
                                               projected_embeddings[index])) for index, train_index in enumerate(self.train_indices)]       
        else:
            projected_embeddings = [np.hstack((self.structural_measures[train_index], projected_embeddings[index])) for index, train_index in enumerate(self.train_indices)]
        # print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        

        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        # train_embeddings_list = [train_embeddings for index, train_embeddings in enumerate(train_embeddings_list) \
        #                          if is_inlier_or_outlier_list[index] == 1]
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) != Neural_Network_Regressor:
            # Projection Mapping from 3D to 1D
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
            # TODO: Change back to train + val
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.val_age_labels)
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels + self.val_age_labels + self.test_age_labels)
        elif type(self.regressor_model) == Neural_Network_Regressor:

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            # age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels + self.val_age_labels)) \
            #     .clone() \
            #     .detach() \
            #     .to(dtype=torch.float32) \
            #     .squeeze()
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")
