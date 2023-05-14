import os, wget, hashlib
from zipfile import ZipFile

#==============================================================#===============================================================#

def download_file(url, out, max_tries = 10, checksum = None, overwrite = 1, verbose = True):
    '''
    Downloads a file using wget.download with some basic
    validation.
    
    ======
    Inputs
    ======
    url : str
        The url to download the file from (as passed to
        wget.download())
    out : str
        The filepath the save the downloaded file to (as
        passed to wget.download())
    max_tries : int
        The maximum number of tries to attempt a download if
        any error(s) occur during download.
    checksum : None or str
        If None, peforms no checksum verification of the file
        at out.
        If str, verifies if this matches the 64-byte BLAKE2
        hash of the file at out (as returned by
        hashlib.blake2b(open(out,'rb').read()).hexdigest()).
    overwrite : int in [0, 1, 2]
        If 0, does not overwrite existing file at out
        regardless of its checksum.
        If 1, overwrites existing file at out iff its checksum
        verification fails.
        If 2, overwrites existing file at out regardless of
        its checksum.
    verbose : bool
        If True, prints status and error messages. If False,
        prints nothing.
    
    =======
    Returns
    =======
    n_tries : int
        Number of unsuccessful tries before sucessful
        download (and checksum verification).
        
    ============
    Dependencies
    ============
    os, wget, hashlib
    '''
    n_tries = 0
    while n_tries < max_tries:
        # Attempt to download file
        try:
            # CONVERT FILEPATH ACCORDING TO OS
            out = os.path.relpath(out)
            
            # MAKE OUTPUT DIRECTORY IF IT DOESN'T ALREADY EXIST
            out_dir = os.path.dirname(out)
            if (not os.path.exists(out_dir)) and len(out_dir) > 0:
                os.makedirs(out_dir)

            # CHECK IF FILE EXISTS AND HANDLE ACCORDINGLY
            if os.path.exists(out):
                if overwrite == 0:
                    if verbose: print(f'{out} already exists. Skipping download...')
                    break
                elif overwrite == 1:
                    if checksum is None:
                        if verbose: print(f'{out} already exists. Skipping download...')
                        break
                    elif hashlib.blake2b(open(out,'rb').read()).hexdigest() == checksum:
                        if verbose: print(f'{out} already exists and matches expected checksum. Skipping download...')
                        break
                    else:
                        if verbose: print(f'{out} already exists but does not match expected checksum. Overwriting...')
                        os.remove(out)
                else: # overwrite == 2
                    os.remove(out)
            
            # DOWNLOAD FILE AT url (WITH PROGRESS BAR IF verbose)
            if verbose:
                print(f'Attempting to download from {url} to {out} (try {n_tries+1}/{max_tries})...')
                wget.download(url,out)
                print()
            else:
                wget.download(url,out,bar=None)

            # CHECK IF FILE DATA MATCHES EXPECTED CHECKSUM
            if checksum is not None:
                assert hashlib.blake2b(open(out,'rb').read()).hexdigest() == checksum
            
            # EXIT WHILE LOOP IF NO ERRORS
            break
        except AssertionError:
            n_tries += 1
            if verbose: print(f'Error: checksum of {out} does not match expected checksum.')
            continue
        except Exception as e:
            n_tries += 1
            if verbose: print(f'Error: {e}')
            continue
        
    return n_tries

#==============================================================#===============================================================#

def unzip(zip_fpath, out_dir = '..', checksum = None, delete_zip = False, verbose = True, **kwargs):
    '''
    Unzips file with zipfile with some basic validation.
    
    ======
    Inputs
    ======
    zip_fpath : str
        The filepath of the zip file.
    out_dir : str
        The directory to extract the zip file's contents to.
    checksum : None or str
        If None, peforms no checksum verification of the file
        at zip_fpath.
        If str, verifies if this matches the 64-byte BLAKE2
        hash of the file at zip_fpath (as returned by
        hashlib.blake2b(open(zip_fpath,'rb').read()).hexdigest())
        before unzipping the file. File is not unzipped if
        checksum does not match.
    delete_zip : bool
        If True, deletes zip file at filepath after successful
        extraction of files. If False, does not delete zip
        file after extraction.
    verbose : bool
        If True, prints status and error messages. If False,
        prints nothing.
    **kwargs : dict
        Additional keyword arguments to ZipFile.extractall
    
    =======
    Returns
    =======
    successful : bool
        If True, zip file was extracted successfully.
        If False, one or more errors occurred during extraction.
        
    ============
    Dependencies
    ============
    ZipFile (from zipfile), hashlib, os
    '''
    # CHECK FILE'S EXISTENCE
    if not os.path.exists(zip_fpath):
        if verbose: print(f'Error: {zip_fpath} does not exist.')
        return False
    
    # VERIFY CHECKSUM
    if (checksum is not None) and (hashlib.blake2b(open(zip_fpath,'rb').read()).hexdigest() != checksum):
        if verbose: print(f'Error: checksum of {zip_fpath} does not match expected checksum.')
        return False
    
    # ATTEMPT TO EXTRACT CONTENTS
    if verbose: print(f'Now unzipping {zip_fpath}...')
    try:
        with ZipFile(zip_fpath) as zip_fh:
            zip_fh.extractall(path = out_dir, **kwargs)
    except Exception as e:
        if verbose: print(f'Error: {e}')
        return False
    
    # DELETE ORIGINAL ZIP FILE IF DESIRED
    if delete_zip:
        if verbose: print(f'Now deleting {zip_fpath}...')
        os.remove(zip_fpath)
        
    return True