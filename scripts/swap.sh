# Based on https://www.digitalocean.com/community/tutorials/how-to-add-swap-space-on-ubuntu-18-04

############################################
# Checking the System for Swap Information #
############################################

# Check if the system has any configured swap
sudo swapon --show

# Verify that there is no active swap using the free utility
free -h

# Check Available Space on the Hard Drive Partition
df -h

########################
# Creating a Swap File #
########################

# Create swapfile '3 Tera bytes', choose whatever size you wish.
sudo fallocate -l 3T /swapfile

# Verify that the correct amount of space was reserved
ls -lh /swapfile

##########################
# Enabling the Swap File #
##########################

# Make the file only accessible to root
sudo chmod 600 /swapfile

# Verify that the file only accessible to root
ls -lh /swapfile

# Mark the file as swap space
sudo mkswap /swapfile

# Enable the swap file
sudo swapon /swapfile

# Verify that the swap is available
sudo swapon --show
free -h

##################################
# Making the Swap File Permanent #
##################################

# Back up the /etc/fstab file in case anything goes wrong
sudo cp /etc/fstab /etc/fstab.bak

# Add the swap file information to the end of your /etc/fstab file
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

#############################
# Tuning your Swap Settings #
#############################

# Adjusting the Swappiness Property #

# Check the current swappiness value
cat /proc/sys/vm/swappiness

# Set the swappiness to a different value by using the sysctl command
sudo sysctl vm.swappiness=10

# Make the value permenant even after restart
echo "vm.swappiness=10" >> /etc/sysctl.conf

# Adjusting the Cache Pressure Setting #

# Check the current value by querying the proc filesystem
cat /proc/sys/vm/vfs_cache_pressure

# Set this to a more conservative setting like 50
sudo sysctl vm.vfs_cache_pressure=50

# Make the value permenant even after restart
echo "vm.vfs_cache_pressure=50" >> /etc/sysctl.conf