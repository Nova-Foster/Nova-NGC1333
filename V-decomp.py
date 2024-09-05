import multiprocessing as mp
import numpy as np
import warnings
from scipy.optimize import curve_fit
from astropy.wcs import wcs
import scipy.optimize
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
import time


def v2ch(v, w):  # convert velocity (km/s) to channel
    x_tempo, y_tempo, v_tempo = w.wcs_pix2world(0, 0, 0, 0)
    x_ch, y_ch, v_ch = w.wcs_world2pix(x_tempo, y_tempo, v * 1000.0, 0)
    v_ch = int(round(float(v_ch), 0))
    return v_ch


def ch2v(ch, w):  # km/s
    x, y, v = w.wcs_pix2world(0, 0, ch, 0)
    return v / 1000.0


def del_header_key(header, keys):  # delete header key
    import copy

    h = copy.deepcopy(header)
    for k in keys:
        try:
            del h[k]
        except:
            pass
    return h


# make header of integrated intensity map
def make_new_hdu_integ(hdu, v_start_wcs, v_end_wcs, w):
    data = hdu.data
    header = hdu.header
    start_ch, end_ch = v2ch(v_start_wcs, w), v2ch(v_end_wcs, w)
    new_data = (
        np.sum(data[start_ch : end_ch + 1], axis=0) * np.abs(header["CDELT3"]) / 1000.0
    )
    header = del_header_key(
        header,
        [
            "CRVAL3",
            "CRPIX3",
            "CRVAL3",
            "CDELT3",
            "CUNIT3",
            "CTYPE3",
            "CROTA3",
            "NAXIS3",
            "PC1_3",
            "PC2_3",
            "PC3_3",
            "PC3_1",
            "PC3_2",
        ],
    )
    header["NAXIS"] = 2
    new_hdu = fits.PrimaryHDU(new_data, header)
    return new_hdu


# make header of integrated intensity map
def make_new_hdu_integ_ch(hdu, v_start_ch, v_end_ch, w):

    data = hdu.data
    header = hdu.header
    new_data = np.sum(data[v_start_ch:v_end_ch], axis=0) * header["CDELT3"] / 1000.0
    header = del_header_key(
        header,
        [
            "CRVAL3",
            "CRPIX3",
            "CRVAL3",
            "CDELT3",
            "CUNIT3",
            "CTYPE3",
            "CROTA3",
            "NAXIS3",
            "PC1_3",
            "PC2_3",
            "PC3_3",
            "PC3_1",
            "PC3_2",
        ],
    )
    header["NAXIS"] = 2
    new_hdu = fits.PrimaryHDU(new_data, header)
    return new_hdu


def gaussian(velocity, A, v_0, sigma):
    return A * np.exp(-((velocity - v_0) ** 2) / (2 * sigma**2))


def gaussian_to_max(current_slice,current_residuals,rms):
    nan_like_slice = np.nan * np.ones_like(current_slice)
    # velocity range to fit to (kms)
    min_expected_vel = -100
    max_expected_vel = 100
    channel_tol = 5                          # originally set to 10, didn't produce very good fits
    # Find V_rad channel at max T
    try:
        channel_max = np.nanargmax(current_slice)
    except ValueError:
        # print(f"Ra index: {ra}, Dec index: {dec} is all Nans. Skipping...")
        return (current_residuals, nan_like_slice, np.nan, np.nan)

    '''if channel_max <= 10:
        # print(f"Max before channel 10: Ra = {ra}, Dec = {dec} - Thread {index}")
        return (current_residuals, nan_like_slice, np.nan, np.nan)'''
    
    
    # Fit gaussian to channel
    try:
        velocity_current_cont = velocity_cont[channel_max - channel_tol : channel_max + channel_tol]
        selected_current_slice = current_slice[channel_max - channel_tol : channel_max + channel_tol]
        vel_peak = np.nanmax(velocity_current_cont)
        flux_peak = np.nanmax(selected_current_slice)
        
        if len(np.isfinite(selected_current_slice)) <= 4:
            # print(f"Fewer than 5 data points: Ra = {ra}, Dec = {dec} - Thread {index}")
            return (current_residuals, nan_like_slice, np.nan, np.nan)

        pop, pcov = curve_fit(gaussian,xdata=velocity_current_cont,ydata=selected_current_slice,p0=[flux_peak, vel_peak, 0.2*5],nan_policy="omit",bounds=([rms, min_expected_vel, 0.2*1], [2*flux_peak, max_expected_vel, 0.2*100]))
    except RuntimeError:
        # print(f"Runtime error: Ra = {ra}, Dec = {dec} - Thread {index}")
        return (current_residuals, nan_like_slice, np.nan, np.nan)
    except IndexError:
        # print(f"Index error: Ra = {ra}, Dec = {dec} - Thread {index}")
        return (current_residuals, nan_like_slice, np.nan, np.nan)
    except scipy.optimize._optimize.OptimizeWarning:
        # print(f"Inditerminate covariance: Ra = {ra}, Dec = {dec} - Thread {index}")
        return (current_residuals, nan_like_slice, np.nan, np.nan)
    except TypeError:
        return(current_residuals,nan_like_slice,np.nan, np.nan)

    # Construct continuus gaussian and subtract from data
    fitted_gauss = gaussian(velocity_cont, pop[0], pop[1], pop[2])
    residuals = current_residuals - fitted_gauss
    errors = np.sqrt(np.diag(pcov))
    return (residuals,fitted_gauss, pop, errors)


def process_chunk(payload):
    chan_low = 20
    chan_high = 200
    # Split the payload array: chunk to be processed, thread ID, dec pixel to start at
    data_local = payload[0]
    index = int(payload[1])
    dec_start = int(payload[2])

    print(f" Thread {index}: dec start: {dec_start}  data shape: {np.shape(data_local)}")
    # Find length of data to be processed to loop through
    shape = np.shape(data_local)
    ra_len = shape[2]
    dec_len = shape[1]

    residuals_local = np.empty((dec_len, ra_len), dtype=np.ndarray)
    model_local = np.empty((dec_len,ra_len),dtype=(np.ndarray))
    threshold_all = np.empty((dec_len, ra_len))
    fit_params_local = np.empty((dec_len, ra_len), dtype=np.ndarray)
    fit_errors_local = np.empty((dec_len, ra_len), dtype=np.ndarray)

    # Loop for each RA pixel
    for ra in range(ra_len):
        print(f"Thread {index}: {(ra/ra_len)*100:.2f}%")
        #print(f'Processing RA: {ra}/{ra_len}, dec length: {dec_len} - Thread {index}')

        # loop for each DEC pixel
        for dec in range(dec_len):
            # Initialize lists instead of arrays
            residuals_current = []
            model_current = []
            fit_params_current = []
            fit_errors_current = []
            fit_worked = True
            prev_velocity=999

            # Get values for current RA and DEC combination
            fitting_slice = data_local[:, dec, ra]
            #fitting_slice[:-5] = 0                                                  # Set last 5 channels to 0, some files have artifacts that could mess with fits
            current_residuals = fitting_slice
            rms = np.sqrt(np.nanmean(fitting_slice[chan_low:chan_high]**2))
            #sigma_clip,_,_ = sigma_clipped_stats(current_slice)
            #threshold = 5 * sigma_clip
            
            threshold = 3 * rms
            threshold_all[dec, ra] = threshold
            
            # Try and fit an arbitrary number of gaussians to data above threshold
            while fit_worked:
                fitting_slice = np.where(
                    fitting_slice >= threshold, fitting_slice, np.nan
                )
                current_residuals, current_model, fit_param, fit_error = gaussian_to_max(fitting_slice,current_residuals,rms)

                if np.isnan(fit_error).any() or fit_error[1]>=0.5*fit_param[1] or prev_velocity==fit_param[1]:
                    current_residuals = np.where(np.isnan(current_residuals),-320,current_residuals)
                    current_model = np.where(np.isnan(current_model),-320,current_model)
                    fit_param, fit_error = (-320, -320, -320), (-320, -320, -320)
                    fit_worked = False
                else:
                    fitting_slice = current_residuals
                    #print(f"Ra pixel: {ra+1}, Dec pixel {dec + 1} worked. V_0 = {fit_param[1]} - Thread {index}")




                # Append data to lists
                #residuals_current.append(current_residuals)
                model_current.append(current_model)
                fit_params_current.append(fit_param)
                fit_errors_current.append(fit_error)
                prev_velocity = fit_param[1]
                


            # Some number of gaussians have been fitted with results stored in x_current
            # Store values in x_local. Will have shape of DEC x RA x inhomogenous (length of x_current)
            residuals_local[dec, ra] = residuals_current
            model_local[dec,ra] = model_current
            fit_params_local[dec, ra] = fit_params_current
            fit_errors_local[dec, ra] = fit_errors_current

    print(f"Thread {index} finished :)")

    
    np.save(f"output/residuals_chunk_{index}", residuals_local,allow_pickle=True)
    np.save(f"output/model_chunk_{index}", model_local, allow_pickle=True)
    np.save(f"output/fitparams_chunk_{index}", fit_params_local,allow_pickle=True)
    np.save(f"output/fiterrors_chunk_{index}", fit_errors_local,allow_pickle=True)
    
    threshold_all = np.where(np.isnan(threshold_all),-320,threshold_all)
    np.save(f"output/threshold_{index}", threshold_all,allow_pickle=True)
    return 0




# Make optmize warning into errors (i.e. covariance can't be estimated so it is a poorfit
warnings.filterwarnings( 
    "error",
    "Covariance of the parameters could not be estimated",
    scipy.optimize._optimize.OptimizeWarning,
)

# Stop all astropy warnings so they don't flood the commandline
warnings.simplefilter("ignore", category=AstropyWarning)
warnings.simplefilter("ignore",category=RuntimeWarning)

species = "13CO"
filepath = f"C:\\Users\\alexf\\Japan internship\\{species}\\ngc1333TP.{species}.cube.valueK.fits"
hdu = fits.open(filepath)[0]
h = hdu.header
data = hdu.data
w = wcs.WCS(h)
data_shape = np.shape(data)
vrad_len = data_shape[0]
ra_len = data_shape[2]
dec_len = data_shape[1]

channel_cont = np.arange(0, vrad_len)
velocity_cont = ch2v(channel_cont, w)


if __name__ == "__main__":
    print("WARNING: Make sure noise channels (in process_chunk), Velocity range to consider (in Gaussian_to_max) and thread count (in main) have been manually set. Any key to continue...")
    input()
    start_time = time.time()
    thread_count=6
    chunks = np.array_split(data, thread_count, axis=1)

    # Remove any un-needed variables from memory
    del data

    ra_start, dec_start, thread_id = [], [], []
    payload = []
    previous_dec_start = 0
    for i in range(thread_count):
            if i != 0:
                shape_current = np.shape(chunks[i])
                current_dec_start = shape_current[1]
                dec_start = np.append(dec_start, current_dec_start + previous_dec_start)

                previous_dec_start = current_dec_start + previous_dec_start

            else:
                dec_start = np.append(dec_start, 0)

            thread_id.append(i)
            payload.append([chunks[i], thread_id[i], dec_start[i]])

    with mp.Pool(thread_count) as p:
        p.map(process_chunk, payload)

    print("Pool is empty :)")
    end_time = time.time()
    print(f"Execution time: { (end_time - start_time) /60:.2f} minutes")