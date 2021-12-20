            
def definde_NWCSAF_variables(missing_labels = None):
    ct_colors = ['#007800', '#000000','#fabefa','#dca0dc',
            '#ff6400', '#ffb400', '#f0f000', '#d7d796',
            '#e6e6e6',  '#c800c8','#0050d7', '#00b4e6',
            '#00f0f0', '#5ac8a0', ]

    ct_indices = [ 1.5, 2.5, 3.5, 4.5, 
               5.5, 6.5, 7.5, 8.5, 
               9.5, 10.5, 11.5, 12.5,
               13.5, 14.5, 15.5]

    ct_labels = ["land", "sea", "snow", "sea ice", 
                 "very low", "low", "middle", "high opaque", 
                 "very high opaque", "fractional", "semi. thin", "semi. mod. thick", 
                 "semi. thick", "semi. above low","semi. above snow"]

    if(missing_labels is not None):
        mis_ind = [ct_labels.index(ml) for ml in missing_labels]
        for ind in sorted(mis_ind, reverse=True):
            for ct_list in [ct_colors, ct_labels, ct_indices]:
                del ct_list[ind]
        

    return ct_colors, ct_indices, ct_labels
