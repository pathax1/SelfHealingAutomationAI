Feature: IKEA Profile Page Simulation

  Scenario: Try accessing profile page
    Given the user is on the IKEA homepage
    When the user clicks on the profile icon
    Then the login page should be displayed
