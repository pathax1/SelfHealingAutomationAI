Feature: IKEA Search Functionality

  Scenario: Search for a product
    Given the user is on the ZARA HP
    When the user searches for "Printed Hoodie"
    Then the results should include item related to "0962/428/898"
