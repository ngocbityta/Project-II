package com.baomoi.service.sorter;

import com.baomoi.model.Item;
import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;

public class TitleSorter implements ISorter<Item> {

    @Override
    public List<Item> sort(List<Item> list) {
        List<Item> sortedList = new ArrayList<>(list);
        sortedList.sort(Comparator.comparing(Item::getArticleTitle));
        return sortedList;
    }
}
