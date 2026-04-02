% Author: Dorian Knight
% Created: April 1st 2026
% Updated: April 2nd 2026
% Description: Generate spectrogram for two species of bird
clear;
clc;


%% Generate all spectrograms for the Black-capped Donacobius and Brown-crested Flycatcher
species1_source = "Black-capped Donacobius raw audio";
species1_destination = "Black-capped Donacobius spectrograms";

species2_source = "Brown-crested Flycatcher raw audio";
species2_destination = "Brown-crested Flycatcher spectrograms";


% generate_spectrograms(species1_source, species1_destination, "Black-capped Donacobius");
generate_spectrograms(species2_source, species2_destination, "Brown-crested Flycatcher");

drive_directory = "D:\CHEMENG4H03_Data";

% Extract all file names in the folder and create a spectrogram for each
% one
function [] = generate_spectrograms(source_path, destination_path, bird_name)

    all_files_bcd = dir(source_path);
    folder_name = all_files_bcd(1).folder;
    num_of_files = length(all_files_bcd);
    if (length(all_files_bcd) > 20)
        num_of_files = 20;
    end

    % Iterate through everyfile and produce the spectrogram
    for index = 3:num_of_files  % First two indeces are directories
    % for index = 3:3
        file = all_files_bcd(index).name;
        file_string = append(folder_name, "\", file);
        disp(['Generating spectrogram for: ', bird_name, file])
        [y, fs] = audioread(file_string);
    
        % Make spectrogram
        [s,f,t] = stft(y, fs);
        sdb = mag2db(abs(s));
        figure = mesh(t,f/1000,sdb);
        cc = max(sdb(:))+[-60 0];
        ax = gca;
        ax.CLim = cc;
        view(2)
        title([bird_name, "Song Spectrogram - file:", file]);
        ylabel("Frequency (kHz)");
        xlabel("Time (seconds)");
        xlim([0 inf]);
        ylim([0 fs/2000]);
        colorbar
    
        % Save spectrogram
        % produced_file = fullfile("D:","CHEMENG4H03_Data",destination_path,append("S1_",string(index)));

        % Determine file prefix
        if (bird_name == "Black-capped Donacobius")
            file_prefix = "S1_";
        elseif (bird_name == "Brown-crested Flycatcher")
            file_prefix = "S2_";
        end

        produced_file = append(destination_path,"\",file_prefix,string(index));
        writematrix(sdb, append(produced_file,".csv"));
        saveas(figure, produced_file, "png"); % Save as PNG
        saveas(figure, produced_file, "fig"); % Save as fig
    
    end
end