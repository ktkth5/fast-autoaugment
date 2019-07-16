# fast-autoaugment

### Installation
1. clone this repository
2. install ray-project (https://ray.readthedocs.io/en/latest/installation.html)
3. pip install -r requirements.txt
4. comment out /ray/tune/schedulers/pbt.py:293.py
    ```
    def chekcpoint():
    
    ```
5. run script
    to search policy:
    - scripts/full_search.sh